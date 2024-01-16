'''
Implementation of the example from page 54ff of Alfio Borzì, The Sequential Quadratic Hamiltonian Method, Solving Optimal Control Problems, CRC Press, Taylor & Francis Group, 2023
Author of the Python code: Laura Gotthardt
Date: 16.01.2024
'''
import numpy as np
from numpy import linalg as LA
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Define some helper functions

def l2_norm_array(u, t):
    '''L_2 norm of an array using trapezoidal rule for integration
        input:
            u (np.array), dim 2, controls for each time point
            t (np.array), dim 1, time points
        return: (float)
    '''
    integrand_1 = u[0]**2
    integrand_2 = u[1]**2
    integral_1 = integrate.trapz(integrand_1, t)
    integral_2 = integrate.trapz(integrand_2, t)
    return np.sqrt(integral_1 + integral_2)

def l1_norm_array(u, t):
    '''L_1 norm of an array using trapezoidal rule for integration
        input:
            u (np.array), dim 2, controls for each time point
            t (np.array), dim 1, time points
        return: (float)
    '''
    integrand_1 = abs(u[0])
    integrand_2 = abs(u[1])
    integral_1 = integrate.trapz(integrand_1, t)
    integral_2 = integrate.trapz(integrand_2, t)
    return integral_1 + integral_2


# Define problem-specific functions

def l(u1, u2):
    '''Define the running cost function used in the Hamiltonian function
        input:
            u1 (float), first dimension of control
            u2 (float), second dimension of control
        return: (float)
    '''
    return (1/2)*(1e-7)*(u1**2 + u2**2) + (1e-7)*(abs(u1) + abs(u2))


def J(t, x, u):
    '''Define the objective function (a linear quadratic regulator problem)
        input:
            t (np.array of floats), time points
            x (np.array of floats), state at the end time (x(1))
            u (np.array of floats), dim 2, controls for each time point
        return: (float)
    '''
    end_cost = 1/2*LA.norm(x - np.array([1, 0, 0]))**2 # end cost, x = x(1) state at end time, [1, 0, 0] desired state at end time
    quad_cost = (1/2)*(1e-7)*l2_norm_array(u, t)**2 # quadratic cost
    lin_cost = (1e-7)*l1_norm_array(u, t) # linear cost
    return end_cost + quad_cost + lin_cost # total cost

def H_epsilon(epsilon, x, u, u_old, p):
    '''Define the Hamiltonian function
        input:
            epsilon (float), penalty parameter for better robustness of the SQH algorithm
            x (np.array of floats), dim 3, state at time t
            u (np.array of floats), dim 2, updated control at time t
            u_old (np.array of floats), dim 2, old control at time t
            p (np.array of floats), dim 3, adjoint state at time t
        return: (float)
    '''
    A = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) # problem specific matrix
    B1 = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]]) # problem specific matrix
    B2 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]) # problem specific matrix
    x_mat = np.array([x]).T  # Convert x to column vector
    p_mat = np.array([p]).T  # Convert p to column vector
    H_general = np.dot(p_mat.T, (A + u[0]*B1 + u[1]*B2) @ x_mat) - l(u[0], u[1]) # General Hamiltonian function
    return H_general - epsilon*((u[0] - u_old[0])**2 + (u[1] - u_old[1])**2) # Hamiltonian function with penalty term (augmented Hamiltonian function)

# Define the SQH algorithm for solving the optimal control problem

def sqh_algorithm():
    '''Implementation of the SQH algorithm
        input: None
        return:
            t_grid (np.array of floats), time points
            x_k (np.array of floats), dim 3, optimal state at each time point
            u_k (np.array of floats), dim 2, optimal control at each time point
            k (int), number of iterations of the SQH algorithm until the stopping criterion is satisfied or the maximum number of iterations is reached
    '''
    # Initialize variables
    A = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) # Problem specific matrix
    B1 = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]]) # Problem specific matrix
    B2 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]) # Problem specific matrix
    x_start = np.array([0, 0, 1])  # Given initial state
    k_max = 80  # Maximum number of iterations
    kappa = 1e-10  # Tolerance for the stopping criterion
    epsilon = 100  # Initial epsilon value, penalty parameter for better robustness of the SQH algorithm
    sigma = 1.2  # Adjustment parameter for epsilon
    eta = 1e-9  # Tolerance parameter for the change in the objective function
    zeta = 0.8  # Adjustment parameter for epsilon
    k = 0 # Iteration counter
    tau = kappa + 1 # Parameter for the relative change in the control, initialize with a value larger than kappa to enter the first while loop
    cond = True # Parameter for the second while loop, initialize with True

    t_0, t_f = 0, 1  # Start and end times for the time domain
    t_grid = np.linspace(t_0, t_f, 100)  # Time points
    u_k = np.zeros((2, 100))  # Initial guess for the control at each time point

    def interpolate_control(t, u_k, t_grid):
        '''Interpolate the control at time t
            input:
                t (float), time point
                u_k (np.array of floats), dim 2, control at each time point
                t_grid (np.array of floats), time points
            return:
                (np.array of floats), dim 2, interpolated control at time t
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
                x (np.array of floats), dim 3, state at time t
                u_k (np.array of floats), dim 2, control at each time point
                t_grid (np.array of floats), time points
            return:
                (np.array of floats), dim 3, state derivative at time t
        '''
        u_current = interpolate_control(t, u_k, t_grid)
        return (A + u_current[0]*B1 + u_current[1]*B2) @ x
    
    sol_x_start = integrate.solve_ivp(fwd_ode_k, [t_0, t_f], x_start, t_eval=t_grid, args=(u_k, t_grid), dense_output=True) # solve the forward ODE using Runge-Kutta method of order (4)5  with initial value x_start
    x_k = sol_x_start.y # State at each time point

    while k < k_max and tau > kappa:
        cond = True
        # Step 1: Solve the adjoint problem with backward integration to get p_k
        def adj_bwd_ode(t, p, u_k, t_grid):
            '''Define the adjoint ODE
                input:
                    t (float), time point
                    p (np.array of floats), dim 3, adjoint state at time t
                    u_k (np.array of floats), dim 2, control at each time point
                    t_grid (np.array of floats), time points
                return:
                    (np.array of floats), dim 3, adjoint state derivative at time t
            '''
            u_current = interpolate_control(t, u_k, t_grid)
            return -(A + u_current[0]*B1 + u_current[1]*B2).T @ p
        
        t_grid_rev = t_grid[::-1] # Reverse the time grid to match the backward time
        p_end = -(x_k[:,-1].reshape((3,)) - np.array([1, 0, 0])) # Initial guess for the adjoint state at the end time
        sol_p = integrate.solve_ivp(adj_bwd_ode, [t_f, t_0], p_end, t_eval=t_grid_rev, args=(u_k, t_grid), dense_output=True)  # Backward integration of the adjoint ODE using Runge-Kutta method of order (4)5 with initial value p_end
        p_k = sol_p.y[:, ::-1]  # Adjoint state at each time point, Reverse the solution to match the forward time
    
        while cond:
            # Step 2: Update the control u_k
            # We evaluate H_epsilon on four control points for each time point seperately and choose u_k_plus_1 equal to that point which maximizes H_epsilon for each time point
            u_1_1 = np.array([]) 
            u_1_2 = np.array([])
            u_2_1 = np.array([])
            u_2_2 = np.array([])
            u_k_plus_1_1 = np.array([]) # 1. dimension of u_k_plus_1
            u_k_plus_1_2 = np.array([]) # 2. dimension of u_k_plus_1

            for i in range(len(p_k[0])):
                # compute the four control points for each time point
                u_1_1_val = max(min(2, (2 * epsilon * u_k[0, i] + np.array([p_k[0, i], p_k[1, i], p_k[2, i]]) @ B1 @ np.array([x_k[0, i], x_k[1, i], x_k[2, i]]) - (1e-7)) / (2 * epsilon + (1e-7))), 0)
                u_1_1 = np.append(u_1_1, u_1_1_val)

                u_1_2_val = max(min(2, (2 * epsilon * u_k[1, i] + np.array([p_k[0, i], p_k[1, i], p_k[2, i]]) @ B2 @ np.array([x_k[0, i], x_k[1, i], x_k[2, i]]) - (1e-7)) / (2 * epsilon + (1e-7))), 0)
                u_1_2 = np.append(u_1_2, u_1_2_val)

                u_2_1_val = max(min(0, (2 * epsilon * u_k[0, i] + np.array([p_k[0, i], p_k[1, i], p_k[2, i]]) @ B1 @ np.array([x_k[0, i], x_k[1, i], x_k[2, i]]) + (1e-7)) / (2 * epsilon + (1e-7))), -2)
                u_2_1 = np.append(u_2_1, u_2_1_val)

                u_2_2_val = max(min(0, (2 * epsilon * u_k[1, i] + np.array([p_k[0, i], p_k[1, i], p_k[2, i]]) @ B2 @ np.array([x_k[0, i], x_k[1, i], x_k[2, i]]) + (1e-7)) / (2 * epsilon + (1e-7))), -2)
                u_2_2 = np.append(u_2_2, u_2_2_val)

                # compute the four H_epsilon values for each time point
                # H_epsilon with (u_1_1, u_1_2)
                h_1_1_val = H_epsilon(epsilon, x_k[:, i], np.array([u_1_1_val, u_1_2_val]), u_k[:, i], p_k[:, i])

                # H_epsilon with (u_1_1, u_2_2)
                h_1_2_val = H_epsilon(epsilon, x_k[:, i], np.array([u_1_1_val, u_2_2_val]), u_k[:, i], p_k[:, i])
 
                # H_epsilon with (u_2_1, u_1_2)
                h_2_1_val = H_epsilon(epsilon, x_k[:, i], np.array([u_2_1_val, u_1_2_val]), u_k[:, i], p_k[:, i])

                # H_epsilon with (u_2_1, u_2_2)
                h_2_2_val = H_epsilon(epsilon, x_k[:, i], np.array([u_2_1_val, u_2_2_val]), u_k[:, i], p_k[:, i])

                # choose u_k_plus_1 equal to that point which maximizes H_epsilon
                if h_1_1_val >= h_1_2_val and h_1_1_val >= h_2_1_val and h_1_1_val >= h_2_2_val:
                    u_k_plus_1_1 = np.append(u_k_plus_1_1, u_1_1_val)
                    u_k_plus_1_2 = np.append(u_k_plus_1_2, u_1_2_val)
                elif h_1_2_val >= h_1_1_val and h_1_2_val >= h_2_1_val and h_1_2_val >= h_2_2_val:
                    u_k_plus_1_1 = np.append(u_k_plus_1_1, u_1_1_val)
                    u_k_plus_1_2 = np.append(u_k_plus_1_2, u_2_2_val)
                elif h_2_1_val >= h_1_1_val and h_2_1_val >= h_1_2_val and h_2_1_val >= h_2_2_val:
                    u_k_plus_1_1 = np.append(u_k_plus_1_1, u_2_1_val)
                    u_k_plus_1_2 = np.append(u_k_plus_1_2, u_1_2_val)
                elif h_2_2_val >= h_1_1_val and h_2_2_val >= h_1_2_val and h_2_2_val >= h_2_1_val:
                    u_k_plus_1_1 = np.append(u_k_plus_1_1, u_2_1_val)
                    u_k_plus_1_2 = np.append(u_k_plus_1_2, u_2_2_val)
                else:
                    print('Error: No maximum found!')
                    break

            u_k_plus_1 = np.array([u_k_plus_1_1, u_k_plus_1_2]) # Updated control at each time point

            # Step 3: Update the state x_k by solving the forward problem using the updated control u_k_plus_1
            def fwd_ode_k_plus_1(t, x, u_k_plus_1, t_grid):
                '''Define the forward ODE
                    input:
                        t (float), time point
                        x (np.array of floats), dim 3, state at time t
                        u_k_plus_1 (np.array of floats), dim 2, updated control at each time point
                        t_grid (np.array of floats), time points
                    return:
                        (np.array of floats), dim 3, state derivative at time t
                '''
                u_current = interpolate_control(t, u_k_plus_1, t_grid)
                return (A + u_current[0]*B1 + u_current[1]*B2) @ x
    
            sol_x = integrate.solve_ivp(fwd_ode_k_plus_1, [t_0, t_f], x_start, t_eval=t_grid, args=(u_k_plus_1, t_grid), dense_output=True) # Solve the forward ODE using Runge-Kutta method of order (4)5  with initial value x_start
            x_k_plus_1 = sol_x.y # Updated state at each time point

            # Step 4: Compute the square of the L^2 norm of the difference between the old and the updated control using the trapezoidal rule
            u_diff = u_k_plus_1 - u_k
            tau = l2_norm_array(u_diff, t_grid)**2 

            # Step 5: Check the change in the objective function using the old and updated states and controls and adjust epsilon
            # The cost functional J is computed using the trapezoidal rule
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
    t_grid, x_k, u_k, k = sqh_algorithm() # Get the solution of the SQH algorithm

    print("optimal control =", u_k)
    print("optimal state =", x_k)
    print("number of iterations =", k)

    # Plot the control and state
    plt.figure(figsize=(12, 4))
    # first subplot for control
    plt.subplot(1, 2, 1)
    plt.plot(t_grid, u_k[0], label='optimal control $u_1$', color='darkorange')
    plt.plot(t_grid, u_k[1], label='optimal control $u_2$', color='coral', linestyle='--')
    plt.xlabel('time t')
    plt.ylabel('control u')
    plt.legend()
    plt.grid(True)
    plt.title('Optimal Control Trajectories')
    # second subplot for state
    plt.subplot(1, 2, 2)
    plt.plot(t_grid, x_k[0], label='optimal state $x_1$', color='forestgreen')
    plt.plot(t_grid, x_k[1], label='optimal state $x_2$', color='mediumseagreen', linestyle='--')
    plt.plot(t_grid, x_k[2], label='optimal state $x_3$', color='olivedrab', linestyle='-.')
    plt.xlabel('time t')
    plt.ylabel('state x')
    plt.legend()
    plt.grid(True)
    plt.title('Optimal State Trajectories')

    plt.show()


if __name__ == '__main__':
    main()