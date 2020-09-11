import sys
import os
import argparse
import time
import h5py
import numpy as np
import math

import log
from config import config

logger = log.get_logger(__name__)

def acq_centroid(settings, flag = 0):
    """
    Calculates S-H spot centroids for each image in data list

    Args:
        flag = 0 for one time centroiding process
        flag = 1 for calibration process
        flag = 2 for test run of closed-loop AO control via Zernikes
        flag = 3 for normal closed-loop AO control via Zernikes
        flag = 4 for normal closed-loop AO control via slopes
        flag = 5 for closed-loop AO control via Zernikes with obscured S-H spots
        flag = 6 for closed-loop AO control via slopes with obscured S-H spots
        flag = 7 for closed-loop AO control via Zernikes with partial correction
        flag = 8 for closed-loop AO control via slopes with partial correction
        flag = 9 for full closed-loop AO control via Zernikes
        flag = 10 for full closed-loop AO control via slopes
        flag = 11 for remote focusing calibration process
    """
    # Open HDF5 file to retrieve calibration images
    data_file = h5py.File('data_info.h5', 'a')

    subgroup_options = {0 : data_file['centroiding_img'],
                        1 : data_file['calibration_img'],
                        2 : data_file['AO_img']['zern_test'],
                        3 : data_file['AO_img']['zern_AO_1'],
                        4 : data_file['AO_img']['slope_AO_1'],
                        5 : data_file['AO_img']['zern_AO_2'],
                        6 : data_file['AO_img']['slope_AO_2'],
                        7 : data_file['AO_img']['zern_AO_3'],
                        8 : data_file['AO_img']['slope_AO_3'],
                        9 : data_file['AO_img']['zern_AO_full'],
                        10 : data_file['AO_img']['slope_AO_full'],
                        11 : data_file['calibration_RF_img']}

    cent_options = {0 : 'real_cent_img', 1 : 'dummy_cent_img'}

    calib_options = {0 : 'real_calib_img', 1 : 'dummy_calib_img'}

    AO_options = {0 : 'real_AO_img', 1 : 'dummy_AO_img'}

    axis_options = {0 : 'dummy_spot_cent_x', 1 : 'dummy_spot_cent_y'}

    if flag == 1:
        if config['DM']['DM_num'] == 0:
            image_num = 2 * config['DM0']['actuator_num']
        elif config['DM']['DM_num'] == 1:
            image_num = 2 * config['DM1']['actuator_num']
    else:
        image_num = 1

    # Get actual search block reference centroid coords
    act_ref_cent_coord = settings['act_ref_cent_coord']
    act_ref_cent_coord_x = settings['act_ref_cent_coord_x']
    act_ref_cent_coord_y = settings['act_ref_cent_coord_y']
    act_SB_coord = settings['act_SB_coord']

    # Get sensor parameters
    sensor_width = settings['sensor_width']
    sensor_height = settings['sensor_height']

    # Get search block outline parameter
    outline_int = config['search_block']['outline_int']

    # Get information of search blocks
    SB_diam = settings['SB_diam']
    SB_rad = settings['SB_rad']

    # Initialise actual S-H spot centroid coords array and sharpness metric array
    act_cent_coord, act_cent_coord_x, act_cent_coord_y, sharpness = (np.zeros(settings['act_ref_cent_num']) for i in range(4))

    # Initialise list for storing S-H spot centroid information for all images in data
    slope_x_list, slope_y_list = ([] for i in range(2))

    # Calculate actual S-H spot centroids for each search block using a dynamic range
    for l in range(image_num):

        if flag == 0:
            image_temp = subgroup_options[flag][cent_options[config['dummy']]][l, :, :]
        elif flag == 1:
            image_temp = subgroup_options[flag][calib_options[config['dummy']]][l, :, :]
        elif flag == 11:
            image_temp = subgroup_options[flag]['real_calib_RF_img'][-1, :, :]
        else:
            image_temp = subgroup_options[flag][AO_options[config['dummy']]][-1, :, :]
        
        # print('Centroiding image {}'.format(l))

        for i in range(settings['act_ref_cent_num']):

            for n in range(config['image']['dynamic_num']):

                # Initialise temporary summing parameters
                sum_x, sum_y, sum_pix = (0 for i in range(3))

                if n == 0:
                    # Get 2D coords of pixels in each search block that need to be summed
                    if settings['odd_pix']:
                        SB_pix_coord_x = np.arange(math.ceil(math.ceil(act_ref_cent_coord_x[i]) - SB_rad), \
                            (math.ceil(math.ceil(act_ref_cent_coord_x[i]) - SB_rad) + settings['SB_diam']))
                        SB_pix_coord_y = np.arange(math.ceil(math.ceil(act_ref_cent_coord_y[i]) - SB_rad), \
                            (math.ceil(math.ceil(act_ref_cent_coord_y[i]) - SB_rad) + settings['SB_diam']))
                    else:
                        SB_pix_coord_x = np.arange(math.ceil(math.ceil(act_ref_cent_coord_x[i]) - SB_rad) + 1, \
                            (math.ceil(math.ceil(act_ref_cent_coord_x[i]) - SB_rad) + settings['SB_diam'] + 1))
                        SB_pix_coord_y = np.arange(math.ceil(math.ceil(act_ref_cent_coord_y[i]) - SB_rad) + 1, \
                            (math.ceil(math.ceil(act_ref_cent_coord_y[i]) - SB_rad) + settings['SB_diam'] + 1))
                else:
                    """
                    Two methods for setting the dynamic range

                    Notes:
                        1) Without thresholding and both doing 5 cycles, Method 2 is 2 - 3 times more effective for uniform noise below 5 
                            (low noise level) and slightly (1 - 2 times) more effective for uniform noise above 7.
                        2) Without thresholding and both doing 5 cycles, Method 2 is 2 times more effective for Gaussian, Method 1 is slightly 
                            more effective for speckle, both are equally effective for Poisson.
                        3) Method 2 is much more stable than Method 1 (error level using Method 1 sometimes double with the same parameters).
                            HOWEVER, when the position of the S-H spot in adjacent search blocks is very close together, problems occur with
                            Method 2 as the search area may incorporate adjacent spots, while Method 1 gives much better accuracy. For spot 
                            centre displacement of (-15, 15), 'noise' = 20, 'threshold' = 0.1, dynamic_num' = 5, positioning error for Method 1
                            is 0.036 (equivalent to 'dynamic_num = 1), while that for Method 2 is 1.369. For large spot centre displacements, 
                            i.e. (-15, 15), positioning error for Method 2 increases with more cycles.                            
                        4) The size of each S-H spot affects centroiding accuracy, the smaller the spot (smaller sigma), the less accurate the
                            centroiding is with all other parameters the same.
                        5) Using Method 2, without thresholding and doing 5 cycles, the average positioning error is around 0.5 for a 
                            uniform noise level of 7 and sigma of 4.                                    
                        6) Using Method 2, without dynamic range and using thresholding value of 0.1, the average positioning error is around 
                            0.002 for a uniform noise level below 28 and sigma of 2 - 4, around 0.3 for a uniform noise level of 29 and sigma
                            of 4, and around 0.6 for a uniform noise level of 30 and sigma of 4. However, the average positioning error increases 
                            rapidly to 0.8 for a uniform noise level of 29 and sigma of 2, and 1.5 for a uniform noise level of 30 and sigma of 2.
                        7) Using Method 2, without dynamic range and using thresholding value of 0.1, the positions (randomness) of each S-H spot 
                            WITHIN the search block does not affect centroiding accuracy, but being on the lines has substantial affect. Using 2 
                            cycles of dynamic range alleviates this affect greatly (to the same accuracy as when all spots are WITHIN the search
                            blocks). In this case, also using thresholding value of 0.1, the average positioning error is around 0.17 for a uniform
                            noise level of 30 and sigma of 4 when all spots are WITHIN the search blocks.                                
                        8) With 180 spots, the time for one centroiding process is around 1.3 s for 1 cycle, 2.5 s for 2 cycles, and 5.5 s for 
                            5 cycles of dynamic range.                                 
                    """
                    # Method 1: Judge the position of S-H spot centroid relative to centre of search block and decrease dynamic range
                    # if act_cent_coord_x[i] > act_ref_cent_coord_x[i]:
                    #     SB_pix_coord_x = SB_pix_coord_x[1:]
                    # elif act_cent_coord_x[i] < act_ref_cent_coord_x[i]:
                    #     SB_pix_coord_x = SB_pix_coord_x[:-1]
                    # else:
                    #     SB_pix_coord_x = SB_pix_coord_x[1:-1]

                    # if act_cent_coord_y[i] > act_ref_cent_coord_y[i]:
                    #     SB_pix_coord_y = SB_pix_coord_y[1:]
                    # elif act_cent_coord_y[i] < act_ref_cent_coord_y[i]:
                    #     SB_pix_coord_y = SB_pix_coord_y[:-1]
                    # else:
                    #     SB_pix_coord_y = SB_pix_coord_y[1:-1]

                    # Method 2: Centre new search area on centroid calculated during previous cycle while shrinking search area at the same time
                    if settings['odd_pix']:
                        SB_pix_coord_x = np.arange(math.ceil(math.ceil(act_cent_coord_x[i]) - SB_rad) + n, \
                            math.ceil(math.ceil(act_cent_coord_x[i]) - SB_rad) + settings['SB_diam'] - n)
                        SB_pix_coord_y = np.arange(math.ceil(math.ceil(act_cent_coord_y[i]) - SB_rad) + n, \
                            math.ceil(math.ceil(act_cent_coord_y[i]) - SB_rad) + settings['SB_diam'] - n)
                    else:
                        SB_pix_coord_x = np.arange(math.ceil(math.ceil(act_cent_coord_x[i]) - SB_rad) + 1 + n, \
                            math.ceil(math.ceil(act_cent_coord_x[i]) - SB_rad) + settings['SB_diam'] + 1 - n)
                        SB_pix_coord_y = np.arange(math.ceil(math.ceil(act_cent_coord_y[i]) - SB_rad) + 1 + n, \
                            math.ceil(math.ceil(act_cent_coord_y[i]) - SB_rad) + settings['SB_diam'] + 1 - n)        

                """
                Calculate actual S-H spot centroids by using centre of gravity (CoG) method
                """

                prev1 = time.perf_counter()
                
                # Crop image within each search area
                image_crop = image_temp[SB_pix_coord_y[0] : SB_pix_coord_y[0] + len(SB_pix_coord_y), \
                    SB_pix_coord_x[0] : SB_pix_coord_x[0] + len(SB_pix_coord_x)]

                # Calculate centroid positions within search area
                if flag in [5, 6, 9, 10]:

                    # Obscured subaperture removal function incorporated
                    # print('Sharpness is:', (image_crop ** 2).sum() / (image_crop.sum()) ** 2)
                    if image_crop.sum() == 0 or (image_crop ** 2).sum() / (image_crop.sum()) ** 2 < config['search_block']['sharp_thres']:
                        
                        # If aperture is obscured, set sum_x, sum_y to 0
                        sum_x, sum_y = (0 for i in range(2))
                        sum_pix = 1
                    else:                     
                        
                        # If aperture isn't obscured, calculate centroid using CoG method
                        xx, yy = np.meshgrid(np.arange(SB_pix_coord_x[0], SB_pix_coord_x[0] + len(SB_pix_coord_x)), \
                            np.arange(SB_pix_coord_y[0], SB_pix_coord_y[0] + len(SB_pix_coord_y)))

                        sum_x = (xx * image_crop).sum()
                        sum_y = (yy * image_crop).sum()
                        sum_pix = image_crop.sum()
                else:

                    # Obscured subaperture removal function not incorporated, calculate centroid using CoG method
                    xx, yy = np.meshgrid(np.arange(SB_pix_coord_x[0], SB_pix_coord_x[0] + len(SB_pix_coord_x)), \
                        np.arange(SB_pix_coord_y[0], SB_pix_coord_y[0] + len(SB_pix_coord_y)))

                    sum_x = (xx * image_crop).sum()
                    sum_y = (yy * image_crop).sum()
                    sum_pix = image_crop.sum()

                # Get actual centroid coordinates
                act_cent_coord_x[i] = sum_x / sum_pix
                act_cent_coord_y[i] = sum_y / sum_pix
                act_cent_coord[i] = math.ceil(act_cent_coord_y[i]) * settings['sensor_width'] + math.ceil(act_cent_coord_x[i])

        if config['dummy']:

            # Calculate average centroid error 
            error_temp = 0

            if flag in [0, 1]:
                error_x = act_cent_coord_x - subgroup_options[flag][axis_options[0]][l, :]
                error_y = act_cent_coord_y - subgroup_options[flag][axis_options[1]][l, :]
            else:
                error_x = act_cent_coord_x - subgroup_options[flag][axis_options[0]][-1, ... ]
                error_y = act_cent_coord_y - subgroup_options[flag][axis_options[1]][-1, ... ]

            for i in range(len(error_x)):
                error_temp += np.sqrt(error_x[i] ** 2 + error_y[i] ** 2)
            error_tot = error_temp / len(error_x)

            # Calculate raw slopes in each dimension
            slope_x = act_cent_coord_x - (act_ref_cent_coord_x.astype(int) + 1)
            slope_y = act_cent_coord_y - (act_ref_cent_coord_y.astype(int) + 1)
        else:

            # Calculate raw slopes in each dimension
            slope_x = act_cent_coord_x - act_ref_cent_coord_x
            slope_y = act_cent_coord_y - act_ref_cent_coord_y

        # Append slopes to list
        slope_x_list.append(slope_x)
        slope_y_list.append(slope_y)

        # print('Act_cent_coord_x:', act_cent_coord_x)
        # print('Act_cent_coord_y:', act_cent_coord_y)
        # print('Error along x axis:', error_x)
        # print('Error along y axis:', error_y)
        # print('Average position error is {}'.format(error_tot))
        # print('Slope along x axis:', slope_x)
        # print('Slope along y axis:', slope_y)

    # Close HDF5 file
    data_file.close()

    prev2 = time.perf_counter()
    # print('Time for centroid calculation process is:', (prev2 - prev1))

    if flag == 1:
        return slope_x_list, slope_y_list
    else:
        return act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x_list, slope_y_list
