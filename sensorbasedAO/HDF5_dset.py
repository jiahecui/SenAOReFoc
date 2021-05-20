import h5py
import math
import time
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
    key_list_1 = ['dummy_cent_img', 'dummy_calib_img', 'dummy_AO_img', 'dummy_spot_cent_x', 'dummy_spot_cent_y', 'dummy_spot_slope_x',\
        'dummy_spot_slope_y', 'dummy_spot_slope', 'dummy_spot_zern_err', 'dummy_spot_slope_err']
    key_list_2 = ['real_cent_img', 'real_calib_img', 'real_calib_RF_img', 'real_AO_img', 'real_spot_slope_x', 'real_spot_slope_y', 'real_spot_slope', \
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
                elif flag == 5 and k == 'real_calib_RF_img':
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
    # f = h5py.File('sensorbasedAO/UnwrappedPhase_IMG_Blastocyte1_Bottom.mat','r')
    # f = h5py.File('sensorbasedAO/UnwrappedPhase_IMG_Blastocyte1_Top.mat','r')
    # f = h5py.File('sensorbasedAO/UnwrappedPhase_IMG_Blastocyte2_Bottom.mat','r')
    # f = h5py.File('sensorbasedAO/UnwrappedPhase_IMG_Blastocyte2_Top.mat','r')
    # f = h5py.File('sensorbasedAO/UnwrappedPhase_IMG_Brain30Gly_Bottom.mat','r')
    f = h5py.File('sensorbasedAO/UnwrappedPhase_IMG_Brain90PBS_Bottom.mat','r')
    # f = h5py.File('sensorbasedAO/UnwrappedPhase_IMG_MouseOocyte.mat','r')
    
    data = f.get('UnwrappedPhase')

    # f = h5py.File('sensorbasedAO/WrappedPhase_IMG_Blastocyte1_Bottom.mat','r')
    # f = h5py.File('sensorbasedAO/WrappedPhase_IMG_Blastocyte1_Top.mat','r')
    # f = h5py.File('sensorbasedAO/WrappedPhase_IMG_Blastocyte2_Bottom.mat','r')
    # f = h5py.File('sensorbasedAO/WrappedPhase_IMG_Blastocyte2_Top.mat','r')
    # f = h5py.File('sensorbasedAO/WrappedPhase_IMG_Brain30Gly_Bottom.mat','r')
    # f = h5py.File('sensorbasedAO/WrappedPhase_IMG_Brain90PBS_Bottom.mat','r')
    # f = h5py.File('sensorbasedAO/WrappedPhase_IMG_MouseOocyte.mat','r')

    # data = f.get('WrappedPhase')

    # Choose working DM along with its parameters
    if config['DM']['DM_num'] == 0:
        pupil_diam = config['search_block']['pupil_diam_0']
    elif config['DM']['DM_num'] == 1:
        pupil_diam = config['search_block']['pupil_diam_1']
    
    # Interpolate to suitable size
    data = np.array(data[130,...]) * config['AO']['lambda'] / (2 * np.pi)  
    mag_fac = pupil_diam / 7.216 * 4
    data_interp = sp.ndimage.zoom(data, mag_fac).T 
    
    # Pad image to same dimension as sensor size
    if np.shape(data_interp)[0] % 2 == 0:
        data_pad = np.pad(data_interp, (math.ceil((settings['sensor_height'] - np.shape(data_interp)[0]) / 2),\
            math.ceil((settings['sensor_width'] - np.shape(data_interp)[1]) / 2)), 'constant', constant_values = (0, 0))
    else:
        data_pad = np.pad(data_interp, (math.ceil((settings['sensor_height'] - np.shape(data_interp)[0]) / 2) - 1,\
            math.ceil((settings['sensor_width'] - np.shape(data_interp)[1]) / 2)), 'constant', constant_values = (0, 0))

    if flag == 0:
        return data_interp
    elif flag == 1:
        return data_pad 
    else:
        return get_slope_from_phase(settings, data_pad)

    """
    Notes regarding amplitude and frequency of aberrations at different sample positions for UnwrappedPhase_IMG_Blastocyte1_Bottom:

        1) 1, 16, 17, 20, 78, 141, 220 is good for demonstrating both AO_zernikes and AO_slopes (small and smooth);
        2) 14, 220 is also good for demonstrating how tip, tilt, defocus are the dominant aberrations such that the DM doesn't need to
            apply any corrections; 
        2) 80 is good for demonstrating AO_zernikes strehl ratio = 0.796, but not reaching 0.8, therefore not breaking from the loop;
        3) 100, 220 is good for demonstrating how the final rms slope error differs between AO_slopes_1 and AO_slopes_3 (1 < 3);
        4) 54 is strongly aberrated in amplitude and gives obscured subapertures;
        5) 58 brings the spots back well, but strehl ratio is limited to 0.758 due to a strong diagonal aberration;
        6) 91, 92 are so strongly aberrated that the spots are scattered all over the place;
        7) 176 is good for demonstrating how AO_slopes brings strehl ratio to 0.82, but AO_zernikes only reaches 0.79;
        8) 99, 100 is good for demonstration of large but smooth amplitude phase variation -> correctable to 0.81;
        9) 102 is good for demonstration of relatively small and smooth phase variation -> only correctable to 0.55 for AO_zernikes,
            0.51 for AO_slopes
        10) Many turbid positions can demonstrate that AO_zernikes_2 is very robust and could bring the strehl ratio back to > 0.81, 
            while AO_slopes_2 can't (strehl < 0.4?): 91, 92, 176, 102, 49, 50, 54, 57
    """