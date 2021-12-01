import h5py
import numpy as np
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
        flag = 0 for data_collection
        flag = 1 for AO_zernikes
        flag = 2 for AO_slopes
        flag = 3 for centroiding
        flag = 4 for calibration
        flag = 5 for remote focusing calibration
    """
    def make_dset(group, name, data):
        """
        Function to create HDF5 dataset with specified shape
        """
        group.create_dataset(name, (0,) + data.shape, maxshape = (3000,) + data.shape, dtype = data.dtype, chunks = (1,) + data.shape)

    # Create dataset shape placeholders
    data_set_img = np.zeros([settings['sensor_height'], settings['sensor_width']])
    data_set_cent = np.zeros(settings['act_ref_cent_num'])
    data_set_slope = np.zeros([settings['act_ref_cent_num'] * 2, 1])
    data_set_zern = np.zeros([config['AO']['recon_coeff_num'], 1])

    # Initialise data set key lists
    key_list = ['real_cent_img', 'real_calib_img', 'real_calib_RF_img', 'real_AO_img', 'real_spot_slope_x', 'real_spot_slope_y', 'real_spot_slope', \
        'real_spot_zern_err', 'real_spot_slope_err']
        
    # Get data file and groups
    data_file = h5py.File('data_info.h5', 'a')
    
    if flag in range(3):
        subgroup_1 = data_file['AO_img'][name]
        subgroup_2 = data_file['AO_info'][name]
    else:
        group = data_file[name]

    # Update datasets in existing subgroups
    for k in key_list:
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
            elif flag == 3 and k == 'real_spot_slope_err':
                make_dset(group, k, data_set_slope)
            elif flag == 3 and k == 'real_spot_zern_err':
                make_dset(group, k, data_set_zern)
            elif flag == 3 and k in {'real_spot_slope_x', 'real_spot_slope_y'}:
                make_dset(group, k, data_set_cent)
            elif flag == 3 and k == 'real_spot_slope':
                make_dset(group, k, data_set_slope)
            elif flag == 4 and k == 'real_calib_img':
                make_dset(group, k, data_set_img)
            elif flag == 5 and k == 'real_calib_RF_img':
                make_dset(group, k, data_set_img)