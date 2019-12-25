import h5py
import numpy as np
import scipy as sp
from scipy import interpolate
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

def get_mat_dset():
    """
    Function to load .mat file image and interpolate to suitable size for analysis
    """
    f = h5py.File('sensorbasedAO/WrappedPhase_IMG_Blastocyte.mat','r')
    data = f.get('WrappedPhase')
    data = np.array(data[-1,...])
    x, y = (np.arange(164) for i in range(2))
    interp_func = sp.interpolate.RectBivariateSpline(x, y, data, kx = 3, ky = 3)
    xx, yy = (np.arange(656) for i in range(2))
    data_interp = interp_func(xx, yy)

    return data_interp