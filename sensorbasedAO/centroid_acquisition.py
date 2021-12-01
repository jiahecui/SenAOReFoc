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
                        2 : data_file['AO_img']['data_collect'],
                        3 : data_file['AO_img']['zern_AO_1'],
                        4 : data_file['AO_img']['slope_AO_1'],
                        5 : data_file['AO_img']['zern_AO_2'],
                        6 : data_file['AO_img']['slope_AO_2'],
                        7 : data_file['AO_img']['zern_AO_3'],
                        8 : data_file['AO_img']['slope_AO_3'],
                        9 : data_file['AO_img']['zern_AO_full'],
                        10 : data_file['AO_img']['slope_AO_full'],
                        11 : data_file['calibration_RF_img']}

    if flag == 1:
        if config['DM']['DM_num'] == 0:
            image_num = 2 * config['DM0']['actuator_num']
        elif config['DM']['DM_num'] == 1:
            image_num = 2 * config['DM1']['actuator_num']
    else:
        image_num = 1

    # Get actual search block reference centroid coords
    act_ref_cent_coord_x = settings['act_ref_cent_coord_x']
    act_ref_cent_coord_y = settings['act_ref_cent_coord_y']

    # Get information of search blocks
    SB_rad = settings['SB_rad']

    # Initialise actual S-H spot centroid coords array and sharpness metric array
    act_cent_coord, act_cent_coord_x, act_cent_coord_y = (np.zeros(settings['act_ref_cent_num']) for i in range(3))

    # Initialise list for storing S-H spot centroid information for all images in data
    slope_x_list, slope_y_list = ([] for i in range(2))

    # Calculate actual S-H spot centroids for each search block using a dynamic range
    for l in range(image_num):

        if flag == 0:
            image_temp = subgroup_options[flag]['real_cent_img'][-1, :, :]
        elif flag == 1:
            image_temp = subgroup_options[flag]['real_calib_img'][l, :, :]
        elif flag == 11:
            image_temp = subgroup_options[flag]['real_calib_RF_img'][-1, :, :]
        else:
            image_temp = subgroup_options[flag]['real_AO_img'][-1, :, :]

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
                    # Centre new search area on centroid calculated during previous cycle while shrinking search area at the same time
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
                
                # Crop image within each search area
                image_crop = image_temp[SB_pix_coord_y[0] : SB_pix_coord_y[0] + len(SB_pix_coord_y), \
                    SB_pix_coord_x[0] : SB_pix_coord_x[0] + len(SB_pix_coord_x)]

                # Calculate centroid positions within search area
                if flag in [5, 6, 9, 10]:

                    # Obscured subaperture removal function incorporated
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

        # Calculate raw slopes in each dimension
        slope_x = act_cent_coord_x - act_ref_cent_coord_x
        slope_y = act_cent_coord_y - act_ref_cent_coord_y

        # Append slopes to list
        slope_x_list.append(slope_x)
        slope_y_list.append(slope_y)

    # Close HDF5 file
    data_file.close()

    if flag == 1:
        return slope_x_list, slope_y_list
    else:
        return act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x_list, slope_y_list
