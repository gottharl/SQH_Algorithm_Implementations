'''
Implementation of the experiment for finding the optimal epsilon value for both versions of the SQH algorithm for solving the example 4.4 from page 99 of Pinch, Enid R., Optimal Control and the Calculus of Variations, Oxford University Press, 1995
Author of the Python code: Laura Gotthardt
Date: 16.01.2024
'''

import SQH_example_3_version_1 as sqh1
import SQH_example_3_version_2 as sqh2
import numpy as np
import matplotlib.pyplot as plt

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

def opt_sqh():
    '''Function to find the optimal epsilon value for both versions of the SQH algorithm for solving example 3. The optimal epsilon value is chosen as the one for which the absolute error for the control and state at the end time is minimal.
        input: None
        return: (np.array), dim 1, error for the control for each epsilon
                (np.array), dim 1, error for the state for each epsilon
    '''
    t_grid = np.linspace(0, 1, 100) # time grid
    error_list_control = np.array([]) # list to store the error for the control for each epsilon
    error_list_state = np.array([]) # list to store the error for the state for each epsilon
    for epsilon in range(1, 71): # loop over all epsilon values from 1 to 70
        # get the approximate solution for the control and state
        x_k, u_k = sqh1.sqh_algorithm(epsilon)[1:3] # SQH version 1, (replace by sqh2.sqh_algorithm(epsilon)[1:3] for SQH version 2)
        # get the analytical solution for the control and state
        u_analyt = analyt_control(t_grid)
        x_analyt = analyt_state(t_grid)
        # calculate the absolute error for the control and state at the end time
        u_error = abs(u_k[-1] - u_analyt[-1])
        x_error = abs(x_k[-1] - x_analyt[-1])
        # append the error to the corresponding list
        error_list_control = np.append(error_list_control, u_error)
        error_list_state = np.append(error_list_state, x_error)
    return error_list_control, error_list_state

def main():
    '''Function to plot the error for the control and state at the end time against epsilon for the chosen version of the SQH algorithm (can be changed in the opt_sqh function)
    '''
    error_list_control, error_list_state = opt_sqh() # get the error for the control and state for each epsilon
    # print the error for the control and state for each epsilon
    print('error for the control at the end time for each epsilon:')
    print(error_list_control)
    print('error for the state at the end time for each epsilon:')
    print(error_list_state)
    epsilon_list = np.linspace(1, 70, 70) # list of all epsilon values
    # plot the error for the control and state at the end time against epsilon
    plt.plot(epsilon_list, error_list_control, label='control')
    plt.plot(epsilon_list, error_list_state, label='state')
    plt.xlabel('epsilon')
    plt.ylabel('error')
    plt.legend()
    plt.grid(True)
    plt.title('Error for the control and state at the end time against epsilon')
    plt.show()

if __name__ == '__main__':
    main()

        
