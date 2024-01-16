'''
Implementation of the experiment for finding the optimal epsilon value for the SQH algorithm for solving the example 5.14 from page 166 of Pinch, Enid R., Optimal Control and the Calculus of Variations, Oxford University Press, 1995
Author of the Python code: Laura Gotthardt
Date: 16.01.2024
'''

import SQH_example_2 as sqh
import numpy as np
import matplotlib.pyplot as plt

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

def opt_sqh():
    '''Function to find the optimal epsilon value for the SQH algorithm for solving example 2. The optimal epsilon value is chosen as the one for which the absolute error for the control and state at the end time is minimal.
        input: None
        return: (np.array), dim 1, error for the control for each epsilon (first dimension)
                (np.array), dim 1, error for the control for each epsilon (second dimension)
                (np.array), dim 1, error for the state for each epsilon (first dimension)
                (np.array), dim 1, error for the state for each epsilon (second dimension)
    '''
    t_grid = np.linspace(0, 3, 1000) # time grid for the given time interval
    error_list_control_1 = np.array([]) # list to store the error for the control for each epsilon (first dimension)
    error_list_control_2 = np.array([]) # list to store the error for the control for each epsilon (second dimension)
    error_list_state_1 = np.array([]) # list to store the error for the state for each epsilon (first dimension)
    error_list_state_2 = np.array([]) # list to store the error for the state for each epsilon (second dimension)
    epsilon = 1
    while epsilon < 11: # loop over all epsilon values from 1 to 10
        # get the approximate solution for the optimal control and state
        x_k, u_k = sqh.sqh_algorithm(epsilon)[1:3]
        # get the analytical solution for the optimal control and state
        u_analyt_1 = np.full_like(t_grid, analyt_control()[0]) # Analytical solution of the control (first dimension)
        u_analyt_2 = np.full_like(t_grid, analyt_control()[1]) # Analytical solution of the control (second dimension)
        x_analyt = analyt_state(t_grid) # Analytical solution of the state
        # compute the absolute error for the control and state at the end time
        u_1_error = abs(u_k[0, -1] - u_analyt_1[-1])
        u_2_error = abs(u_k[1, -1] - u_analyt_2[-1])
        x_error_1 = abs(x_k[0, -1] - x_analyt[0, -1])
        x_error_2 = abs(x_k[1, -1] - x_analyt[1, -1])
        # append the error to the corresponding list
        error_list_control_1 = np.append(error_list_control_1, u_1_error)
        error_list_control_2 = np.append(error_list_control_2, u_2_error)
        error_list_state_1 = np.append(error_list_state_1, x_error_1)
        error_list_state_2 = np.append(error_list_state_2, x_error_2)
        epsilon += 0.5 # increase epsilon by 0.5
    return error_list_control_1, error_list_control_2, error_list_state_1, error_list_state_2

def main():
    '''Function to plot the error for the optimal control and state at the end time against epsilon
    '''
    error_list_control_1, error_list_control_2, error_list_state_1, error_list_state_2 = opt_sqh() # get the error for the optimal control and state for each epsilon
    # print the error for the optimal control and state for each epsilon
    print('Error for the control at the end time for each epsilon:')
    print(error_list_control_1)
    print(error_list_control_2)
    print('Error for the state at the end time for each epsilon:')
    print(error_list_state_1)
    print(error_list_state_2)
    epsilon_list = np.linspace(1, 10, 20) # list of epsilon values
    # plot the error for the control and state at the end time against epsilon
    plt.figure(figsize=(12, 4))
    # first subplot: error for the control at the end time
    plt.subplot(1, 2, 1)
    plt.semilogy(epsilon_list, error_list_control_1, '.-', label='control $u_1$', color= 'springgreen')
    plt.semilogy(epsilon_list, error_list_control_2, '.--', label='control $u_2$', color= 'black')
    plt.xlabel('epsilon')
    plt.ylabel('error')
    plt.legend()
    plt.grid(True)
    plt.title('Error for the control at the end time against epsilon')
    # second subplot: error for the state at the end time
    plt.subplot(1, 2, 2)
    plt.semilogy(epsilon_list, error_list_state_1, '.-', label='state $x_1$', color= 'springgreen')
    plt.semilogy(epsilon_list, error_list_state_2, '.--', label='state $x_2$', color= 'black')
    plt.xlabel('epsilon')
    plt.ylabel('error')
    plt.legend()
    plt.grid(True)
    plt.title('Error for the state at the end time against epsilon')
    plt.show()

if __name__ == '__main__':
    main()

        
