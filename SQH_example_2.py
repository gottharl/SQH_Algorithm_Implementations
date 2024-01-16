'''
Implementation of the SQH algorithm for solving the example 5.14 from page 166 of Pinch, Enid R., Optimal Control and the Calculus of Variations, Oxford University Press, 1995
Author of the Python code: Laura Gotthardt
Date: 16.01.2024
'''
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define helper functions

def l2_norm_array(u, t):
    '''L_2 norm of an array using the trapezoidal rule for integration
        input:
            u (np.array), dim 2, control for each time point
            t (np.array), dim 1, time points
        return: (float)
    '''
    integrand_1 = u[0]**2
    integrand_2 = u[1]**2
    integral_1 = integrate.trapz(integrand_1, t)
    integral_2 = integrate.trapz(integrand_2, t)
    return np.sqrt(integral_1 + integral_2)

# Define problem-specific functions

def J(t, x, u):
    '''Objective function, cost functional
        input:
            t (np.array), dim 1, time points
            x (np.array), dim 2, state at the end time point
            u (np.array), dim 2, control for each time point
        return: (float)
    '''
    integrand = 0.5 * (u[0]**2 + u[1]**2)
    integral = integrate.trapz(integrand, t)
    return integral + 0.5 * (x[0]**2 + x[1]**2)

def H_epsilon(epsilon, u, u_old, p):
    '''Define the Hamiltonian function
        input:
            epsilon (float), penalty parameter for better robustness of the SQH algorithm
            u (np.array), dim 2, updated control at time t
            u_old (np.array), dim 2, old control at time t
            p (np.array), dim 2, adjoint state at time t
        return: (float) 
    '''
    H_general = - 0.5 * (u[0]**2 + u[1]**2) + p[0] * u[0] + p[1] * u[1] # general Hamiltonian function
    return H_general - epsilon*((u[0] - u_old[0])**2 + (u[1] - u_old[1])**2) # augmented Hamiltonian function

def H_epsilon_1deriv(epsilon, u, u_old, p):
    '''Gradient of the Hamiltonian function w.r.t. u, needed for the maximization problem in the SQH algorithm
        input:
            epsilon (float), penalty parameter for better robustness of the SQH algorithm
            u (np.array), dim 2, updated control at time t
            u_old (np.array), dim 2, old control at time t
            p (np.array), dim 2, adjoint state at time t
        return: (np.array)
    '''
    return p - u - epsilon * 2 * (u - u_old)

# Define the SQH algorithm for solving the optimal control problem

def sqh_algorithm(epsilon):
    '''Implementation of the SQH algorithm
        input: epsilon (float), penalty parameter for better robustness of the SQH algorithm
        return:
            t_grid (np.array), time points
            x_k (np.array), dim 2, optimal state at each time point
            u_k (np.array), dim 2, optimal control at each time point
            k (int), number of iterations of the SQH algorithm until convergence or maximum number of iterations is reached
    '''
    # Initialize variables
    x_start = np.array([0.5, 2]) # Given initial state
    k_max = 80  # Maximum number of iterations
    kappa = 1e-10  # Tolerance parameter for the stopping criterion
    sigma = 1.1  # Adjustment parameter for epsilon
    eta = 1e-8  # Tolerance parameter for  the change in the objective function
    zeta = 0.8  # Adjustment parameter for epsilon
    k = 0 # Iteration counter
    tau = kappa + 1 # Parameter for the relative change in the control, initialize with a value larger than kappa to enter the first while loop
    cond = True # Parameter for the second while loop, initialize with True

    t_0, t_f = 0, 3  # Start and end time for the time domain
    t_grid = np.linspace(t_0, t_f, 1000)  # Time points
    u_k = np.zeros((2, 1000))  # Initial guess for the control at each time point

    def interpolate_control(t, u_k, t_grid):
        '''Interpolate the control at time t
            input:
                t (float), time point
                u_k (np.array), dim 2, control at each time point
                t_grid (np.array), time points
            return:
                (np.array), dim 2, interpolated control at time t
        '''
        u1_interpolated = np.interp(t, t_grid, u_k[0])
        u2_interpolated = np.interp(t, t_grid, u_k[1])
        return np.array([u1_interpolated, u2_interpolated])
    
    # Main SQH algorithm loop

    # Step 0: Solve the forward problem to get x_0, the zeroth iterate of x
    def fwd_ode_k(t, x, u_k, t_grid):
        '''Define the forward ODE
            input:
                t (float), time point
                x (np.array), dim 2, state at time t
                u_k (np.array), dim 2, control at each time point
                t_grid (np.array), time points
            return:
                (np.array), dim 2, state derivative at time t
        '''
        u_current = interpolate_control(t, u_k, t_grid)
        return u_current
    
    sol_x_start = integrate.solve_ivp(fwd_ode_k, t_span=[t_0, t_f], y0=x_start, t_eval=t_grid, args=(u_k, t_grid), dense_output=True) # Solve the forward ODE using Runge-Kutta method of order (4)5  with initial value x_start
    x_k = sol_x_start.y # State at each time point

    while k < k_max and tau > kappa:
        cond = True
        # Step 1: Solve the adjoint problem  with backward integration to get p_k
        def adj_bwd_ode(t, p):
            '''Define the adjoint ODE
                input:
                    t (float), time point
                    p (np.array), dim 2, adjoint state at time t
                return:
                    (np.array), dim 2, adjoint state derivative at time t
            '''
            return 0
        
        t_grid_rev = t_grid[::-1] # Reverse the time grid to match the backward time
        p_end = np.array(-x_k[:, -1].reshape((2,))) # Initial value for the adjoint state at the end time
        sol_p = integrate.solve_ivp(adj_bwd_ode, t_span=[t_f, t_0], y0=p_end, t_eval=t_grid_rev, dense_output=True) # Backward integration solving the adjoint ODE using Runge-Kutta method of order (4)5  with initial value p_end
        p_k = sol_p.y[:, ::-1] # Adjoint state at each time point, reverse the solution to match the forward time

        while cond:
            # Step 2: Update the control u_k by solving the optimization problem H_epsilon(x_k, u_k_plus_1, u_k, p_k) = max_u H_epsilon(x_k, u, u_k, p_k) for each time point separately
            u_k_plus_1_1 = np.array([])  # Initialize u_k_plus_1, first dimension
            u_k_plus_1_2 = np.array([])  # Initialize u_k_plus_1, second dimension

            for i in range(len(t_grid)):

                def objective(u):
                    '''Objective function (Hamiltonian) to be maximized (negative because optimize.minimize function is used)
                        input:
                            u (np.array), dim 2, control at time t_grid[i]
                        return: (float)
                    '''
                    return -H_epsilon(epsilon, u, u_k[:, i], p_k[:, i])
                
                def gradient(u):
                    '''Gradient of the objective function (Hamiltonian)
                        input:
                            u (np.array), dim 2, control at time t_grid[i]
                        return: (np.array), dim 2, gradient of the objective function
                    '''
                    return -H_epsilon_1deriv(epsilon, u, u_k[:, i], p_k[:, i])

                result = minimize(objective, u_k[:, i], method='BFGS', jac=gradient) # minimize the objective function using the Broyden–Fletcher–Goldfarb–Shanno algorithm 
                if result.success:
                    u_k_plus_1_1 = np.append(u_k_plus_1_1, result.x[0])
                    u_k_plus_1_2 = np.append(u_k_plus_1_2, result.x[1])
            
            u_k_plus_1 = np.array([u_k_plus_1_1, u_k_plus_1_2]) # Updated control at each time point

            # Step 3: Update the state x_k by solving the forward problem using the updated control u_k_plus_1
            def fwd_ode_k_plus_1(t, x, u_k_plus_1, t_grid):
                '''Define the forward ODE
                    input:
                        t (float), time point
                        x (np.array), dim 2, state at time t
                        u_k_plus_1 (np.array), dim 2, updated control at each time point
                        t_grid (np.array), time points
                    return:
                        (np.array), dim 2, state derivative at time t
                '''
                u_current = interpolate_control(t, u_k_plus_1, t_grid)
                return u_current
            
            sol_x_k_plus_1 = integrate.solve_ivp(fwd_ode_k_plus_1, t_span=[t_0, t_f], y0=x_start, t_eval=t_grid, args=(u_k_plus_1, t_grid), dense_output=True) # Solve the forward ODE using Runge-Kutta method of order (4)5  with initial value x_start
            x_k_plus_1 = sol_x_k_plus_1.y # Updated state at each time point

            # Step 4: Compute the square of the L^2 norm of the difference between the old and the updated control using the trapezoidal rule
            u_diff = u_k_plus_1 - u_k
            tau = l2_norm_array(u_diff, t_grid)**2

            # Step 5: Check the change in the objective function using the old and updated states and controls and adjust epsilon
            # the cost functional J is computed using the trapezoidal rule
            if J(t_grid, x_k_plus_1[:, -1], u_k_plus_1) - J(t_grid, x_k[:, -1], u_k) > -eta * tau:
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

    return t_grid, x_k, u_k, k # Return the time grid, the optimal state, the optimal control and the number of iterations

def main():
    '''Main function, prints the solution and plots the optimal control and state'''

    t_grid, x_k, u_k, k = sqh_algorithm(epsilon = 1.5) # Get the solution of the SQH algorithm for the optimal epsilon = 1.5 (optimal for this example, see experiment_2.py)
    
    # Plot the control and state with the analytical solution of the control
    def analyt_control():
        '''Analytical solution of the optimal control
            input: None
            return: (np.array), control at each time point, constant for each dimension
        '''
        return np.array([-0.5/4, -2/4])
    
    def analyt_state(t):
        '''Analytical solution of the optimal state
            input:
                t (np.array), dim 1, time points
            return: (np.array), dim 2, state at each time point
        '''
        return np.array([-t * 0.125 + 0.5, -t * 0.5 + 2])
    
    u_analyt_1 = np.full_like(t_grid, analyt_control()[0]) # Analytical solution of the first dimension of the control
    u_analyt_2 = np.full_like(t_grid, analyt_control()[1]) # Analytical solution of the second dimension of the control
    x_analyt = analyt_state(t_grid) # Analytical solution of the state
    absolute_error_control_1 = abs(u_k[0] - u_analyt_1) # Absolute error of the first dimension of the control
    absolute_error_control_2 = abs(u_k[1] - u_analyt_2) # Absolute error of the second dimension of the control
    absolute_error_state_1 = abs(x_k[0] - x_analyt[0]) # Absolute error of the first dimension of the state
    absolute_error_state_2 = abs(x_k[1] - x_analyt[1]) # Absolute error of the second dimension of the state

    print("optimal control =", u_k)
    print("optimal state =", x_k)
    print("number of iterations =", k)

    plt.figure(figsize=(12, 4))
    # first subplot for control
    plt.subplot(1, 2, 1)
    plt.plot(t_grid, u_analyt_1, label='Analytical solution $u_1$', color='yellow')
    plt.plot(t_grid, u_k[0], label='SQH solution $u_1$', linestyle='--', color='teal')
    plt.plot(t_grid, u_analyt_2, label='Analytical solution $u_2$', color='springgreen')
    plt.plot(t_grid, u_k[1], label='SQH solution $u_2$', linestyle=':', color= 'black')
    plt.xlabel('time t')
    plt.ylabel('control u')
    plt.ylim(-1, 0)
    plt.legend()
    plt.grid(color= '#DDDDDD')
    plt.title('Optimal Control Trajectories')
    # second subplot for state
    plt.subplot(1, 2, 2)
    plt.plot(t_grid, x_analyt[0], label='Analytical solution $x_1$', color='yellow')
    plt.plot(t_grid, x_k[0], label='SQH solution $x_1$', linestyle='--', color='teal')
    plt.plot(t_grid, x_analyt[1], label='Analytical solution $x_2$', color='springgreen')
    plt.plot(t_grid, x_k[1], label='SQH solution $x_2$', linestyle=':', color= 'black')
    plt.xlabel('time t')
    plt.ylabel('state x')
    plt.legend()
    plt.grid(color= '#DDDDDD')
    plt.title('Optimal State Trajectories')
    plt.show()

    # plot the absolute error of the control and state
    plt.figure(figsize=(12, 4))
    # first subplot for control
    plt.subplot(1, 2, 1)
    plt.plot(t_grid, absolute_error_control_1, label='Absolute error in $u_1$', color='yellow')
    plt.plot(t_grid, absolute_error_control_2, label='Absolute error in $u_2$', linestyle= '--', color='teal')
    plt.xlabel('time t')
    plt.ylabel('absolute error')
    plt.legend()
    plt.grid(color= '#DDDDDD')
    plt.title('Absolute Error of the Control Trajectories')
    # second subplot for state
    plt.subplot(1, 2, 2)
    plt.semilogy(t_grid, absolute_error_state_1, label='Absolute error in $x_1$', color='springgreen')
    plt.semilogy(t_grid, absolute_error_state_2, label='Absolute error in $x_2$', linestyle= ':', color='black')
    plt.xlabel('time t')
    plt.ylabel('absolute error')
    plt.legend()
    plt.grid(color= '#DDDDDD')
    plt.title('Absolute Error of the State Trajectories')
    plt.show()

if __name__ == '__main__':
    main()