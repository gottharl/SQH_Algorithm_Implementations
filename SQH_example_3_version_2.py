'''
Second version of the implementation of the example 4.4 from page 99 of Pinch, Enid R., Optimal Control and the Calculus of Variations, Oxford University Press, 1995
Author of the Python code: Laura Gotthardt
Date: 16.01.2024
'''
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# Define helper functions

def l2_norm_array(u, t):
    '''L_2 norm of an array using trapezoidal rule for integration
        input:
            u (np.array), dim 1, control for each time point
            t (np.array), dim 1, time points
        return: (float)
    '''
    integrand = u**2
    integral = integrate.trapz(integrand, t)
    return np.sqrt(integral)

# Define problem-specific functions

def J(t, x, u, rho):
    '''Objective function
        input:
            t (np.array of floats), dim 1, time points
            x (np.array of floats), dim 1, state at each time point
            u (np.array of floats), dim 1, control at each time point
        return: (float)
    '''
    integrand = 0.5 * (3 * x**2 + u**2)
    integral = integrate.trapz(integrand, t)
    return integral + rho * abs(2 - x[-1])**2 

def H_epsilon(epsilon, x, u, u_old, p):
    '''Define the Hamiltonian function
        input:
            epsilon (float), penalty parameter for better robustness of the SQH algorithm
            x (np.array of floats), dim 1, state at time t
            u (np.array of floats), dim 1, updated control at time t
            u_old (np.array of floats), dim 1, old control at time t
            p (np.array of floats), dim 1, adjoint state at time t
        return: (float)
    '''
    H_general = p * (-x + u) - 0.5 * (3 * x**2 + u**2) # General Hamiltonian function
    return H_general - epsilon * (u - u_old)**2 # Augmented Hamiltonian function

def H_epsilon_1deriv(epsilon, u, u_old, p):
    '''Gradient of the Hamiltonian function w.r.t. u, needed for the maximization problem in the SQH algorithm
        input:
            epsilon (float), penalty parameter for better robustness of the SQH algorithm
            u (np.array), dim 1, updated control at time t
            u_old (np.array), dim 1, old control at time t
            p (np.array), dim 1, adjoint state at time t
        return: (float)
    '''
    return -u + p - 2 * epsilon * (u - u_old)

# Define the SQH algorithm for solving the optimal control problem

def sqh_algorithm(epsilon):
    '''Implementation of the SQH algorithm
        input: epsilon (float), penalty parameter for better robustness of the SQH algorithm
        return:
            t_grid (np.array of floats), time points
            x_k (np.array of floats), dim 1, optimal state at each time point
            u_k (np.array of floats), dim 1, optimal control at each time point
            k (int), number of iterations of the SQH algorithm until convergence or maximum number of iterations is reached
    '''
    # Initialize variables
    x_start = np.array([0]) # Given initial state
    k_max = 1000  # Maximum number of iterations
    kappa_2 = 1e-3 # Tolerance for the end condition
    k = 0 # Iteration counter
    sigma = 1.1  # Adjustment parameter for epsilon
    eta = 1e-3  # Tolerance parameter for  the change in the objective function
    zeta = 0.7  # Adjustment parameter for epsilon
    cond = True # Parameter for the second while loop, initialize with True
    rho = 1 # Penalty parameter for the objective function
    
    def end_cond(x_k):
        '''End condition for the SQH algorithm. Relaative error of the state at the end time point
            input:
                x_k (np.array of floats), dim 1, state at each time point
            return: (float)
        '''
        return abs(2 - x_k[-1]) / 2

    t_0, t_f = 0, 1  # Start and end time for the time domain
    t_grid = np.linspace(t_0, t_f, 100)  # Time points
    u_k = np.ones((100))  # Initial guess for the control at each time point
    
    def interpolate_control(t, u_k, t_grid):
        '''Interpolate the control at time t
            input:
                t (float), time point
                u_k (np.aray of floats), dim 1, control at each time point
                t_grid (np.array of floats), time points
            return: (float), interpolated control at time t
        '''
        return np.interp(t, t_grid, u_k)
    
    def interpolate_state(t, x_k, t_grid):
        '''Interpolate the state at time t
            input:
                t (float), time point
                x_k (np.aray of floats), dim 1, state at each time point
                t_grid (np.array of floats), time points
            return: (float), interpolated state at time t
        '''
        return np.interp(t, t_grid, x_k)
    
    
    # Main SQH algorithm loop

    # Step 0: Solve the forward problem to get x_0, the zeroth iterate of x
    def fwd_ode_k(t, x, u_k, t_grid):
        '''Define the forward ODE
            input:
                t (float), time point
                x (np.array of floats), dim 1, state at time t
                u_k (np.array of floats), dim 1, control at each time point
                t_grid (np.array of floats), time points
            return:
                (np.array of floats), dim 1, state derivative at time t
        '''
        u_current = interpolate_control(t, u_k, t_grid)
        return -x + u_current
    
    sol_x_start = integrate.solve_ivp(fwd_ode_k, t_span=[t_0, t_f], y0=x_start, t_eval=t_grid, args=(u_k, t_grid), dense_output=True) # Solve the forward ODE using Runge-Kutta method of order (4)5  with initial value x_start
    x_k = sol_x_start.y[0] # State at each time point

    while k < k_max and end_cond(x_k) > kappa_2:
        cond = True
        # Step 1: Solve the adjoint problem with backward integration to get p_k
        def adj_bwd_ode(t, p, x_k, t_grid):
            '''Define the adjoint ODE
                input:
                    t (float), time point
                    p (np.array of floats), dim 1, adjoint state at time t
                    x_k (np.array of floats), dim 1, state at each time point
                    t_grid (np.array of floats), time points
                return:
                    (np.array of floats), dim 1, adjoint state derivative at time t
            '''
            x_current = interpolate_state(t, x_k, t_grid)
            return p + 3 * x_current
        
        t_grid_rev = t_grid[::-1] # Reverse the time grid to match the backward time
        p_end = np.array([2 * rho * abs(x_k[-1] - 2)]) # Initial value for the adjoint state at the end time
        sol_p = integrate.solve_ivp(adj_bwd_ode, t_span=[t_f, t_0], y0=p_end, t_eval=t_grid_rev, args=(x_k, t_grid), dense_output=True) # Backward integration solving the adjoint ODE using Runge-Kutta method of order (4)5  with initial value p_end
        p_k_s = sol_p.y[0]
        p_k = p_k_s[::-1]  # Adjoint state at each time point, Reverse the solution to match the forward time

        while cond:
            # Step 2: Update the control u_k by solving the optimization problem H_epsilon(x_k, u_k_plus_1, u_k, p_k) = max_u H_epsilon(x_k, u, u_k, p_k)
            u_k_plus_1 = np.copy(u_k)  # Initialize u_k_plus_1

            for i in range(len(t_grid)):
            # optimize for each time point separately
                def objective(u):
                    '''Objective function (Hamiltonian) to be maximized (negative because optimize.minimize function is used)
                        input:
                            u (float), control at time t_grid[i]
                        return: (float)
                    '''
                    return -H_epsilon(epsilon, x_k[i], u, u_k[i], p_k[i])
                
                def gradient(u):
                    '''Gradient of the objective function (Hamiltonian) w.r.t. u
                        input:
                            u (float), control at time t_grid[i]
                        return: (float)
                    '''
                    return -H_epsilon_1deriv(epsilon, u, u_k[i], p_k[i])

                result = minimize(objective, u_k[i], method='BFGS', jac=gradient) # Minimize the objective function using the Broyden–Fletcher–Goldfarb–Shanno algorithm
                if result.success:
                    u_k_plus_1[i] = result.x[0] # Update the control at time t_grid[i]

            # Step 3: Update the state x_k by solving the forward problem using the updated control u_k_plus_1
            def fwd_ode_k_plus_1(t, x, u_k_plus_1, t_grid):
                '''Define the forward ODE
                    input:
                        t (float), time point
                        x (np.array of floats), dim 1, state at time t
                        u_k_plus_1 (np.array of floats), dim 1, updated control at each time point
                        t_grid (np.array of floats), time points
                    return:
                        (np.array of floats), dim 1, state derivative at time t
                '''
                u_current = interpolate_control(t, u_k_plus_1, t_grid)
                return -x + u_current
        
            sol_x = integrate.solve_ivp(fwd_ode_k_plus_1, t_span=[t_0, t_f], y0=x_start, t_eval=t_grid, args=(u_k_plus_1, t_grid), dense_output=True) # Solve the forward ODE using Runge-Kutta method of order (4)5  with initial value x_start
            x_k_plus_1 = sol_x.y[0] # Updated state at each time point

            # Step 4: Compute the square of the L^2 norm of the difference between the old and the updated control using the trapezoidal rule
            u_diff = u_k_plus_1 - u_k
            tau = l2_norm_array(u_diff, t_grid)**2

            # Step 5: Check the change in the objective function using the old and updated states and controls and adjust epsilon
            # the cost functional J is computed using the trapezoidal rule
            if J(t_grid, x_k_plus_1, u_k_plus_1, rho) - J(t_grid, x_k, u_k, rho) > -eta * tau:
                epsilon *= sigma # Increase epsilon
                print("eps =", epsilon)
                # Repeat from step 2 with the new epsilon without updating the control
            else:
                epsilon *= zeta # Decrease epsilon
                print("eps =", epsilon)
                cond = False
                # Continue with the updated control

        # Step 6: Update the iteration count, control and state
        k += 1
        u_k = u_k_plus_1
        x_k = x_k_plus_1
        print("tau =", tau)
        rho += 1 # Increase the penalty parameter for the objective function
        print("rho =", rho)

        print("end condition:", end_cond(x_k))
        

    return t_grid, x_k, u_k, k # return the time grid, the optimal state, the optimal control and the number of iterations

def main():
    '''Main function, prints the solution and the running time and plots the optimal control and state'''
    start_time = time.time() # Start time of the SQH algorithm
    t_grid, x_k, u_k, k = sqh_algorithm(epsilon=36) # Run the SQH algorithm for the optimal epsilon = 36 (optimal for this example, see experiment_3.py) and get the optimal state and control and the number of iterations
    end_time = time.time() # End time of the SQH algorithm

    print("optimal control =", u_k)
    print("optimal state =", x_k)
    print("number of iterations =", k)
    print("running time:", end_time - start_time)

    # Plot the control and state with the analytical solution for comparison and the absolute error
    def analyt_control(t):
        '''Analytical solution of the optimal control
            input:
                t (np.array of floats), dim 1, time points
            return: (np.array of floats), dim 1, control at each time point
        '''
        return 2 * (2 * np.cosh(2 * t) + np.sinh(2 * t)) / np.sinh(2)
    
    def analyt_state(t):
        '''Analytical solution of the optimal state
            input:
                t (np.array of floats), dim 1, time points
            return: (np.array of floats), dim 1, state at each time point
        '''
        return np.sinh(2 * t) * (2 / np.sinh(2))
    
    u_analyt = analyt_control(t_grid) # Analytical solution of the control
    x_analyt = analyt_state(t_grid) # Analytical solution of the state
    absolute_error_control = abs(u_k - u_analyt) # Absolute error of the control
    absolute_error_state = abs(x_k - x_analyt) # Absolute error of the state


    plt.figure(figsize=(12, 4))
    # first subplot for control
    plt.subplot(1, 2, 1)
    plt.plot(t_grid, u_analyt, label='Analytical solution', color='greenyellow')
    plt.plot(t_grid, u_k, label='SQH solution', color='black', linestyle= ':')
    plt.xlabel('time t')
    plt.ylabel('control u')
    plt.legend()
    plt.grid(color= '#DDDDDD')
    plt.title('Optimal Control Trajectory')
    # second subplot for state
    plt.subplot(1, 2, 2)
    plt.plot(t_grid, x_analyt, label='Analytical solution of the state', color='greenyellow')
    plt.plot(t_grid, x_k, label='SQH solution of the state', color='black', linestyle= ':')
    plt.xlabel('time t')
    plt.ylabel('state x')
    plt.legend()
    plt.grid(color= '#DDDDDD')
    plt.title('Optimal State Trajectory')
    plt.show()

    # Plot the absolute error of the control and state
    plt.figure(figsize=(12, 4))
    plt.plot(t_grid, absolute_error_control, label='Absolute error of the control', color='midnightblue', linestyle= '--')
    plt.plot(t_grid, absolute_error_state, label='Absolute error of the state', color='black', linestyle= ':')
    plt.xlabel('time t')
    plt.ylabel('absolute error')
    plt.legend()
    plt.grid(color= '#DDDDDD')
    plt.title('Absolute Error')
    plt.show()


if __name__ == '__main__':
    main()