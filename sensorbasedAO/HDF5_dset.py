import h5py
import math
import numpy as np
import scipy as sp
from scipy import ndimage
from config import config
from common import get_slope_from_phase

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
    data_set_zern = np.zeros([config['AO']['recon_coeff_num'], 1])

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
                elif flag in [0, 1, 2] and k in {'dummy_spot_cent_x', 'dummy_spot_cent_y'}:
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

def get_mat_dset(settings, flag = 1):
    """
    Function to load .mat file image and interpolate to suitable size for analysis, includes option for whether to get S-H spots 
    from phase data

    Args:
        flag = 0 - retrieve unpadded phase image
        flag = 1 - retrieve padded phase image
        flag = 2 - retrieve S-H spot slopes from phase data
    """
    # Retrieve phase data from .mat file
    f = h5py.File('sensorbasedAO/WrappedPhase_IMG_Blastocyte.mat','r')
    data = f.get('WrappedPhase')

    # Interpolate to suitable size
    data = np.array(data[-4,...]) * config['AO']['lambda'] / (2 * np.pi)  # -4
    mag_fac = config['search_block']['pupil_diam'] / 7.216 * 4
    data_interp = sp.ndimage.zoom(data, mag_fac).T

    # Pad image to same dimension as sensor size
    data_pad = np.pad(data_interp, (math.ceil((settings['sensor_height'] - np.shape(data_interp)[0]) / 2),\
        math.ceil((settings['sensor_width'] - np.shape(data_interp)[1]) / 2)), 'constant', constant_values = (0, 0))

    if flag == 0:
        return data_interp
    elif flag == 1:
        return data_pad 
    else:
        return get_slope_from_phase(settings, data_pad)