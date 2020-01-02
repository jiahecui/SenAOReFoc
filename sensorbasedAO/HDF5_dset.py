import h5py
import numpy as np
import scipy as sp
from scipy import ndimage
from config import config

def dset_append(group, name, data):
    """
    Function to append an element to end of dataset
    """
    shape = list(group[name].shape)
    shape[0] += 1
    group[name].resize(shape)
    group[name][-1, ...] = data

def get_dset(settings, name, flag = 0):
    """
    Function to update datasets in existing subgroups

    Args:
        flag = 0 for AO_zernikes_test
        flag = 1 for AO_zernikes
        flag = 2 for AO_slopes
        flag = 3 for centroiding
        flag = 4 for calibration
    """
    def make_dset(group, name, data):
        """
        Function to create HDF5 dataset with specified shape
        """
        group.create_dataset(name, (0,) + data.shape, maxshape = (config['DM']['actuator_num'] * 2,) + data.shape, dtype = data.dtype)

    # Create dataset shape placeholders
    data_set_img = np.zeros([settings['sensor_width'], settings['sensor_height']])
    data_set_cent = np.zeros(settings['act_ref_cent_num'])
    data_set_slope = np.zeros([settings['act_ref_cent_num'] * 2, 1])
    data_set_zern = np.zeros([config['AO']['control_coeff_num'], 1])

    # Initialise data set key lists
    key_list_1 = ['dummy_cent_img', 'dummy_calib_img', 'dummy_AO_img', 'dummy_spot_cent_x', 'dummy_spot_cent_y', 'dummy_spot_slope_x',\
        'dummy_spot_slope_y', 'dummy_spot_slope', 'dummy_spot_zern_err', 'dummy_spot_slope_err']
    key_list_2 = ['real_cent_img', 'real_calib_img', 'real_AO_img', 'real_spot_slope_x', 'real_spot_slope_y', 'real_spot_slope', \
        'real_spot_zern_err', 'real_spot_slope_err']
        
    # Get data file and groups
    data_file = h5py.File('data_info.h5', 'a')
    
    if flag in range(3):
        subgroup_1 = data_file['AO_img'][name]
        subgroup_2 = data_file['AO_info'][name]
    else:
        group = data_file[name]

    # Update datasets in existing subgroups
    if config['dummy']:
        for k in key_list_1:
            if flag in range(3):
                if k in subgroup_1:
                    del subgroup_1[k]
                elif k in subgroup_2:
                    del subgroup_2[k]
                if k == 'dummy_AO_img':
                    make_dset(subgroup_1, k, data_set_img)
                elif flag == 0 and k in {'dummy_spot_cent_x', 'dummy_spot_cent_y'}:
                    make_dset(subgroup_1, k, data_set_cent)
                elif flag == 2 and k in {'dummy_spot_slope_err'}:
                    make_dset(subgroup_2, k, data_set_slope)
                elif k in {'dummy_spot_zern_err'}:
                    make_dset(subgroup_2, k, data_set_zern)
                elif k in {'dummy_spot_slope_x', 'dummy_spot_slope_y'}:
                    make_dset(subgroup_2, k, data_set_cent)
                elif k in {'dummy_spot_slope'}:
                    make_dset(subgroup_2, k, data_set_slope)
            else:
                if k in group:
                    del group[k]
                if flag == 3 and k == 'dummy_cent_img':
                    make_dset(group, k, data_set_img)
                elif flag == 4 and k == 'dummy_calib_img':
                    make_dset(group, k, data_set_img)
                elif k in {'dummy_spot_cent_x', 'dummy_spot_cent_y'}:
                    make_dset(group, k, data_set_cent)
    else:
        for k in key_list_2:
            if flag in range(3):
                if k in subgroup_1:
                    del subgroup_1[k]
                elif k in subgroup_2:
                    del subgroup_2[k]
                if k == 'real_AO_img':
                    make_dset(subgroup_1, k, data_set_img)
                elif flag == 2 and k in {'real_spot_slope_err'}:
                    make_dset(subgroup_2, k, data_set_slope)
                elif k in {'real_spot_zern_err'}:
                    make_dset(subgroup_2, k, data_set_zern)
                elif k in {'real_spot_slope_x', 'real_spot_slope_y'}:
                    make_dset(subgroup_2, k, data_set_cent)
                elif k in {'real_spot_slope'}:
                    make_dset(subgroup_2, k, data_set_slope)
            else:
                if k in group:
                    del group[k]
                if flag == 3 and k == 'real_cent_img':
                    make_dset(group, k, data_set_img)
                elif flag == 4 and k == 'real_calib_img':
                    make_dset(group, k, data_set_img)

def get_mat_dset(settings, get_spots = 1):
    """
    Function to load .mat file image and interpolate to suitable size for analysis, includes option for whether to get S-H spots 
    from phase data
    """
    # Retrieve phase data from .mat file
    f = h5py.File('sensorbasedAO/WrappedPhase_IMG_Blastocyte.mat','r')
    data = f.get('WrappedPhase')

    # Interpolate to suitable size
    data = np.array(data[-1,...]) * config['AO']['lambda'] / (2 * np.pi)
    data_interp = sp.ndimage.zoom(data, 4).T

    # Pad image to same dimension as sensor size
    data_interp = np.pad(data_interp, ((settings['sensor_height'] - np.shape(data_interp)[0]) // 2,\
        (settings['sensor_width'] - np.shape(data_interp)[1]) // 2), 'constant', constant_values = (0, 0))

    # Get S-H spots from phase data if required
    if get_spots:

        # Initialise array to store S-H spot centroid position within each search block
        x_slope, y_slope = (np.zeros(settings['act_ref_cent_num']) for i in range(2))

        # Get S-H spot centroid position within each search block
        for i in range(settings['act_ref_cent_num']):

            # Get 2D coords of pixels in each search block that need to be summed
            if settings['odd_pix']:
                SB_pix_coord_x = np.arange(int(settings['act_ref_cent_coord_x'][i]) - settings['SB_rad'] + 1, \
                    int(settings['act_ref_cent_coord_x'][i]) + settings['SB_rad'] - 1)
                SB_pix_coord_y = np.arange(int(settings['act_ref_cent_coord_y'][i]) - settings['SB_rad'] + 1, \
                    int(settings['act_ref_cent_coord_y'][i]) + settings['SB_rad'] - 1)
            else:
                SB_pix_coord_x = np.arange(int(settings['act_ref_cent_coord_x'][i]) - settings['SB_rad'] + 2, \
                    int(settings['act_ref_cent_coord_x'][i]) + settings['SB_rad'] - 1)
                SB_pix_coord_y = np.arange(int(settings['act_ref_cent_coord_y'][i]) - settings['SB_rad'] + 2, \
                    int(settings['act_ref_cent_coord_y'][i]) + settings['SB_rad'] - 1)

            # Initialise instance variables for calculating wavefront tilt within each search block
            a_x, a_y = (np.zeros(len(SB_pix_coord_x)) for i in range(2))

            # Get wavefront tilt of each row and column within each search block
            for j in range(len(SB_pix_coord_x)):
                a_x[j] = np.polyfit(SB_pix_coord_x, data_interp[int(round(SB_pix_coord_y[j])), int(round(SB_pix_coord_x[0])) : \
                    int(round(SB_pix_coord_x[0])) + len(SB_pix_coord_x)], 1)[0] 
                a_y[j] = np.polyfit(SB_pix_coord_y, data_interp[int(round(SB_pix_coord_y[0])) : int(round(SB_pix_coord_y[0])) + \
                    len(SB_pix_coord_y), int(round(SB_pix_coord_x[j]))], 1)[0] 

            # Calculate average wavefront tilt within each search block
            a_x_ave = -np.mean(a_x) / settings['pixel_size']
            a_y_ave = -np.mean(a_y) / settings['pixel_size']

            # Calculate S-H spot centroid position along x and y axis
            x_slope[i] = a_x_ave * config['lenslet']['lenslet_focal_length'] / settings['pixel_size']
            y_slope[i] = a_y_ave * config['lenslet']['lenslet_focal_length'] / settings['pixel_size']

    if get_spots:
        return x_slope, y_slope
    else:
        return data_interp