import sys
import os
import argparse
import time
import h5py
import numpy as np

import log
from config import config

logger = log.get_logger(__name__)

def acq_centroid(settings, flag = 0):
    """
    Calculates S-H spot centroids for each image in data list

    Args:
        flag = 0 for one time centroiding process
        flag = 1 for calibration process
        flag = 2 for closed-loop AO process
    """
    # Open HDF5 file to retrieve calibration images
    data_file = h5py.File('data_info.h5', 'a')

    if flag == 1:
        image_num = 2 * config['DM']['actuator_num']
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

    # Initialise actual S-H spot centroid coords array
    act_cent_coord, act_cent_coord_x, act_cent_coord_y = (np.zeros(settings['act_ref_cent_num']) for i in range(3))

    # Initialise list for storing S-H spot centroid information for all images in data
    slope_x_list, slope_y_list = ([] for i in range(2))

    prev1 = time.perf_counter()

    # Calculate actual S-H spot centroids for each search block using a dynamic range
    for l in range(image_num):

        if flag == 0:
            
            if config['dummy']:
                image_temp = data_file['centroiding_img']['dummy_cent_img'][l, :, :]
            else:
                image_temp = data_file['centroiding_img']['real_cent_img'][l, :, :]

        elif flag == 1:

            if config['dummy']:
                image_temp = data_file['calibration_img']['dummy_calib_img'][l, :, :]
            else:
                image_temp = data_file['calibration_img']['real_calib_img'][l, :, :]

        elif flag == 2:

            if config['dummy']:
                image_temp = data_file['AO_img']['dummy_AO_img'][-1, ... ]
            else:
                image_temp = data_file['AO_img']['real_AO_img'][-1, ... ]
        
        # print('Centroiding image {}'.format(l))

        for i in range(settings['act_ref_cent_num']):

            for n in range(config['image']['dynamic_num']):

                # Initialise temporary summing parameters
                sum_x = 0
                sum_y = 0
                sum_pix = 0

                if n == 0:
                    # Get 2D coords of pixels in each search block that need to be summed
                    if settings['odd_pix']:
                        SB_pix_coord_x = np.arange(act_ref_cent_coord_x[i] - SB_rad + 1, \
                            act_ref_cent_coord_x[i] + SB_rad - 1)
                        SB_pix_coord_y = np.arange(act_ref_cent_coord_y[i] - SB_rad + 1, \
                            act_ref_cent_coord_y[i] + SB_rad - 1)
                    else:
                        SB_pix_coord_x = np.arange(act_ref_cent_coord_x[i] - SB_rad + 2, \
                            act_ref_cent_coord_x[i] + SB_rad - 1)
                        SB_pix_coord_y = np.arange(act_ref_cent_coord_y[i] - SB_rad + 2, \
                            act_ref_cent_coord_y[i] + SB_rad - 1)
                else:
                    """
                    Two methods for setting the dynamic range

                    Notes:
                        1) Without thresholding and both doing 5 cycles, Method 2 is 2 - 3 times more effective for uniform noise below 5 
                            (low noise level) and slightly (1 - 2 times) more effective for uniform noise above 7.
                        2) Without thresholding and both doing 5 cycles, Method 2 is 2 times more effective for Gaussian, Method 1 is slightly 
                            more effective for speckle, both are equally effective for Poisson.
                        3) Method 2 is much more stable than Method 1 (error level using Method 1 sometimes double with the same parameters).
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
                    SB_pix_coord_x = np.arange(act_cent_coord_x[i] - SB_rad + 1 + n, \
                        act_cent_coord_x[i] + SB_rad - 1 - n)
                    SB_pix_coord_y = np.arange(act_cent_coord_y[i] - SB_rad + 1 + n, \
                        act_cent_coord_y[i] + SB_rad - 1 - n)

                # if i == 0:
                #     print('SB_pixel_coord_x_{}_{}: {}'.format(i, n, SB_pix_coord_x))
                #     print('SB_pixel_coord_y_{}_{}: {}'.format(i, n, SB_pix_coord_y))
                #     print('Length of pixel coord along x axis for cycle {}: {}'.format(n, len(SB_pix_coord_x)))
                #     print('Length of pixel coord along y axis for cycle {}: {}'.format(n, len(SB_pix_coord_y)))

                # Calculate actual S-H spot centroids by doing weighted sum
                for j in range(len(SB_pix_coord_y)):
                    for k in range(len(SB_pix_coord_x)):
                        sum_x += image_temp[int(SB_pix_coord_y[j]), int(SB_pix_coord_x[k])] * int(SB_pix_coord_x[k])
                        sum_y += image_temp[int(SB_pix_coord_y[j]), int(SB_pix_coord_x[k])] * int(SB_pix_coord_y[j])
                        sum_pix += image_temp[int(SB_pix_coord_y[j]), int(SB_pix_coord_x[k])]

                act_cent_coord_x[i] = sum_x / sum_pix
                act_cent_coord_y[i] = sum_y / sum_pix
                act_cent_coord[i] = int(act_cent_coord_y[i]) * sensor_width + int(act_cent_coord_x[i])                          

        if config['dummy']:
            # Calculate average centroid error 
            error_temp = 0

            if flag == 0:
                error_x = act_cent_coord_x - data_file['centroiding_img']['dummy_spot_cent_x'][l, :]
                error_y = act_cent_coord_y - data_file['centroiding_img']['dummy_spot_cent_y'][l, :]
            elif flag == 1:
                error_x = act_cent_coord_x - data_file['calibration_img']['dummy_spot_cent_x'][l, :]
                error_y = act_cent_coord_y - data_file['calibration_img']['dummy_spot_cent_y'][l, :]
            elif flag == 2:
                error_x = act_cent_coord_x - data_file['AO_img']['dummy_spot_cent_x'][-1, ... ]
                error_y = act_cent_coord_y - data_file['AO_img']['dummy_spot_cent_y'][-1, ... ]

            for i in range(len(error_x)):
                error_temp += np.sqrt(error_x[i] ** 2 + error_y[i] ** 2)
            error_tot = error_temp / len(error_x)

        # Calculate raw slopes in each dimension
        slope_x = act_cent_coord_x - act_ref_cent_coord_x
        slope_y = act_cent_coord_y - act_ref_cent_coord_y

        # Append slopes to list
        slope_x_list.append(slope_x)
        slope_y_list.append(slope_y)

        # print('Act_cent_coord_x:', act_cent_coord_x)
        # print('Act_cent_coord_y:', act_cent_coord_y)
        # print('Act_cent_coord:', act_cent_coord)
        # print('Error along x axis:', error_x)
        # print('Error along y axis:', error_y)
        # print('Average position error {}: {}'.format(l + 1, error_tot))
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
