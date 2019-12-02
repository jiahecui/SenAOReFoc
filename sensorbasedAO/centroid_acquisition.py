import sys
import os
import argparse
import time
import numpy as np

import log
from config import config

logger = log.get_logger(__name__)

def acq_centroid(settings, spot_cent_x, spot_cent_y, data = None):
    """
    Calculates S-H spot centroids for each image in data list
    """
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
    act_cent_coord_list, act_cent_coord_x_list, act_cent_coord_y_list, slope_x_list, slope_y_list = ([] for i in range(5))
    prev1 = time.perf_counter()

    # Calculate actual S-H spot centroids for each search block using a dynamic range
    for l in range(len(data)):

        image_temp = data[l]

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

        # Calculate average centroid error 
        error_temp = 0
        error_x = act_cent_coord_x - spot_cent_x
        error_y = act_cent_coord_y - spot_cent_y
        for i in range(len(error_x)):
            error_temp += np.sqrt(error_x[i] ** 2 + error_y[i] ** 2)
        error_tot = error_temp / len(error_x)

        # Calculate raw slopes in each dimension
        slope_x = act_cent_coord_x - act_ref_cent_coord_x
        slope_y = act_cent_coord_y - act_ref_cent_coord_y
        # Append relevent parameters to list
        act_cent_coord_list.append(act_cent_coord)
        act_cent_coord_x_list.append(act_cent_coord_x)
        act_cent_coord_y_list.append(act_cent_coord_y)
        slope_x_list.append(slope_x)
        slope_y_list.append(slope_y)

        # print('Act_cent_coord_x:', act_cent_coord_x)
        # print('Act_cent_coord_y:', act_cent_coord_y)
        # print('Act_cent_coord:', act_cent_coord)
        # print('Error along x axis:', error_x)
        # print('Error along y axis:', error_y)
        print('Average position error:', error_tot)
        # print('Slope along x axis:', slope_x)
        # print('Slope along y axis:', slope_y)

    prev2 = time.perf_counter()
    print('Time for one centroiding process is:', (prev2 - prev1))

    return act_cent_coord_list, act_cent_coord_x_list, act_cent_coord_y_list, slope_x_list, slope_y_list
