import numpy as np
from config import config

def inf(xx, yy, xc, yc, j, act_diam):
    """
    Generates the value of Gaussian distribution influence function at each pixel

    Args:
        xx - x value meshgrid
        yy - y value meshgrid
        xc - actuator x position array
        yc - actuator y position array
        j - actuator index, starts from 0
        act_diam - actuator diameter in pixels
    """
    # Choose working DM along with its parameters
    if config['DM']['DM_num'] == 0:
        coupling_fac = config['DM0']['coupling_fac']
    elif config['DM']['DM_num'] == 1:
        coupling_fac = config['DM1']['coupling_fac']

    # Calculate influence function value at each pixel introduced by actuator j
    Fn = np.exp(np.log(coupling_fac) * ((xx - xc[j]) ** 2 + (yy - yc[j]) ** 2) / act_diam ** 2)

    return Fn

def inf_diff(xx, yy, xc, yc, j, act_diam, x_flag = True):
    """
    Generates the slopes of modeled Gaussian distribution influence function for simulation of DM control matrix
    
    Args:
        xx - x value array
        yy - y value array
        xc - actuator x position array
        yc - actuator y position array
        j - actuator index, starts from 0
        act_diam - actuator diameter in pixels
        x_flag - flag for differentiating relative to x or y
    """
    # Choose working DM along with its parameters
    if config['DM']['DM_num'] == 0:
        coupling_fac = config['DM0']['coupling_fac']
    elif config['DM']['DM_num'] == 1:
        coupling_fac = config['DM1']['coupling_fac']

    # Initialise instance variables
    count = 0
    diff_sum = 0

    # Calculate averaged derivative of the jth actuator influence function
    for y in yy:
        for x in xx:

            count += 1

            # Calculate exponential factor
            expo = np.exp(np.log(coupling_fac) * ((x - xc[j]) ** 2 + (y - yc[j]) ** 2) / act_diam ** 2)

            # Calculate x or y derivative of Gaussian distribution influence function and get final result
            if x_flag:
                dFn_dx = 2 * np.log(coupling_fac) / act_diam ** 2 * (x - xc[j])
                dFn = dFn_dx * expo
            else:
                dFn_dy = 2 * np.log(coupling_fac) / act_diam ** 2 * (y - yc[j])
                dFn = dFn_dy * expo

            # Sum derivative for each element in search block
            diff_sum += dFn
    
    # Divide sum of derivative by total number of counts:
    dFn_ave = diff_sum / count

    return dFn_ave