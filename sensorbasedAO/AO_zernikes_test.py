from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import logging
import sys
import os
import argparse
import time
import h5py
from scipy import io
from tifffile import imsave
import numpy as np
import scipy as sp

import log
from config import config
from HDF5_dset import dset_append, get_dset
from image_acquisition import acq_image
from centroid_acquisition import acq_centroid
from gaussian_inf import inf
from common import fft_spot_from_phase
from zernike_phase import zern_phase
from reflectance_process import reflect_process

logger = log.get_logger(__name__)

class AO_Zernikes_Test(QObject):
    """
    Tests closed-loop AO via zernikes / slopes for ideal / DM control matrix generated Zernike modes (iterates through each mode)
    """
    start = Signal()
    write = Signal()
    done = Signal()
    error = Signal(object)
    image = Signal(object)
    message = Signal(object)
    info = Signal(object)

    # def __init__(self, sensor, mirror, scanner, settings):
    def __init__(self, sensor, mirror, settings):

        # Get search block settings
        self.SB_settings = settings['SB_info']

        # Get mirror settings
        self.mirror_settings = settings['mirror_info']

        # Get AO settings
        self.AO_settings = settings['AO_info']

        # Get sensor instance
        self.sensor = sensor

        # Get mirror instance
        self.mirror = mirror

        # Get scanner instance
        # self.scanner = scanner

        # Initialise AO information parameter
        self.AO_info = {'zern_test': {}}

        # Initialise zernike coefficient array
        self.zern_coeff = np.zeros([config['AO']['control_coeff_num'], 1])

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_0']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_1']

        # Initialise array to store accurate Zernike mode voltages
        self.zern_volts = np.zeros([self.actuator_num, config['AO']['control_coeff_num'] - 2, config['zern_test']['incre_num']])
        
        super().__init__()

    def phase_calc(self, voltages):
        """
        Calculates phase profile introduced by DM
        """
        # Get meshgrid of coordinates within phase image
        coord_xx, coord_yy = self.get_coord_mesh(self.SB_settings['sensor_width'])

        # Get boolean pupil mask
        pupil_mask = self.get_pupil_mask(coord_xx, coord_yy)

        # Calculate Gaussian distribution influence function value introduced by each actuator at each pixel
        delta_phase = np.zeros([self.SB_settings['sensor_width'], self.SB_settings['sensor_width']])

        for i in range(self.actuator_num):
            delta_phase += inf(coord_xx, coord_yy, self.mirror_settings['act_pos_x'], self.mirror_settings['act_pos_y'],\
                i, self.mirror_settings['act_diam']) * voltages[i]

        delta_phase = delta_phase * pupil_mask

        # print('Max and min values in delta_phase are: {} um, {} um'.format(np.amax(delta_phase), np.amin(delta_phase)))

        return delta_phase

    def get_coord_mesh(self, image_diam):
        """
        Retrieves coordinate meshgrid for calculations over a full image
        """
        coord_x, coord_y = (np.arange(int(-image_diam / 2), int(-image_diam / 2) + image_diam) for i in range(2))
        coord_xx, coord_yy = np.meshgrid(coord_x, coord_y)

        return coord_xx, coord_yy

    def get_pupil_mask(self, coord_xx, coord_yy):
        """
        Retrieves boolean pupil mask for round aperture
        """
        pupil_mask = np.sqrt((coord_xx * self.SB_settings['pixel_size']) ** 2 + \
            (coord_yy * self.SB_settings['pixel_size']) ** 2) <= self.pupil_diam * 1e3 / 2
        
        return pupil_mask

    @Slot(object)
    def run0(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Run closed-loop AO correction for each generated zernike mode aberration via Zernike control for all 'control_coeff_num' modes 
            and multiple amplitudes of incremental steps 
            """
            # Save calibration slope values and zernike influence function to file
            sp.io.savemat('zern_gen_det_cor/calib_slope_x.mat', dict(calib_slope_x = self.mirror_settings['calib_slope_x']))
            sp.io.savemat('zern_gen_det_cor/calib_slope_y.mat', dict(calib_slope_y = self.mirror_settings['calib_slope_y']))
            sp.io.savemat('zern_gen_det_cor/inf_matrix_zern.mat', dict(inf_matrix_zern = self.mirror_settings['inf_matrix_zern']))

            # Get number of Zernike modes to generate
            zern_num = config['AO']['control_coeff_num'] - 2

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            self.message.emit('\nProcess started for closed-loop AO via Zernikes...')

            prev1 = time.perf_counter()

            for n in range(config['zern_test']['run_num']):

                # Run closed-loop control for each zernike mode aberration
                for k in range(config['zern_test']['incre_num']):

                    # Initialise AO information parameter
                    self.AO_info = {'zern_test': {}}

                    # Create new datasets in HDF5 file to store closed-loop AO data and open file
                    get_dset(self.SB_settings, 'zern_test', flag = 0)
                    data_file = h5py.File('data_info.h5', 'a')
                    data_set_1 = data_file['AO_img']['zern_test']
                    data_set_2 = data_file['AO_info']['zern_test']

                    # Initialise array to store initial and final detected value of each generated Zernike mode 
                    # and RMS Zernike value / Strehl ratio considering recon_coeff_num
                    self.det_cor_zern_coeff = np.zeros([zern_num * 2, zern_num])
                    self.det_cor_rms_zern = np.zeros([zern_num * 2, 1])
                    self.det_cor_strehl = np.zeros([zern_num * 2, 1])

                    # Determine the amplitude to be generated for each Zernike mode
                    zern_amp_gen = config['zern_test']['incre_amp'] * (k + 1)

                    # Determine initial loop gain for generation of each Zernike mode
                    if zern_amp_gen <= 0.2:
                        loop_gain_gen = 0.2
                    elif zern_amp_gen > 0.2:
                        loop_gain_gen = 0.3

                    for j in range(zern_num):

                        print('On amplitude {} Zernike mode {}'.format(k + 1, j + 3))

                        for i in range(self.AO_settings['loop_max'] + 1):
                            
                            if self.loop:
                                
                                try:

                                    # Update mirror control voltages
                                    if i == 0:

                                        if not config['dummy']:

                                            # Generate one Zernike mode on DM for correction each time
                                            self.zern_coeff[j + 2, 0] = zern_amp_gen

                                            # Run closed-loop to generate a precise amount of Zernike modes using DM
                                            for m in range(config['zern_test']['loop_max_gen']):

                                                if m == 0:

                                                    voltages[:] = config['DM']['vol_bias']

                                                else:

                                                    # Update control voltages
                                                    voltages -= loop_gain_gen * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                        [:,:config['AO']['control_coeff_num']], (zern_array_det[:config['AO']['control_coeff_num']] - self.zern_coeff)))

                                                # Send values vector to mirror
                                                self.mirror.Send(voltages)

                                                # Wait for DM to settle
                                                time.sleep(config['DM']['settling_time'])
                                            
                                                # Acquire S-H spots using camera
                                                AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                                AO_image = np.mean(AO_image_stack, axis = 2)

                                                # Image thresholding to remove background
                                                AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                                AO_image[AO_image < 0] = 0
                                                self.image.emit(AO_image)

                                                # Append image to list
                                                dset_append(data_set_1, 'real_AO_img', AO_image)

                                                # Calculate centroids of S-H spots
                                                act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2)
                                                act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                                # Draw actual S-H spot centroids on image layer
                                                AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                                self.image.emit(AO_image)

                                                # Take tip\tilt off
                                                slope_x -= np.mean(slope_x)
                                                slope_y -= np.mean(slope_y)

                                                # Concatenate slopes into one slope matrix
                                                slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                                # Get detected zernike coefficients from slope matrix
                                                zern_array_det = np.dot(self.mirror_settings['conv_matrix'], slope)

                                                # print('Detected amplitude of mode {} is {} um'.format(m + 3, zern_array_det[j + 2, 0]))

                                                if abs(zern_array_det[j + 2, 0] - zern_amp_gen) / zern_amp_gen <= 0.075 or m == config['zern_test']['loop_max_gen'] - 1:
                                                    if config['zern_test']['save_voltages']:
                                                        self.zern_volts[:, j, k] = voltages
                                                    break
                                        else:

                                            voltages[:] = config['DM']['vol_bias']                              
                                    else:

                                        voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                            [:,:config['AO']['control_coeff_num']], zern_err[:config['AO']['control_coeff_num']]))

                                        # print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

                                    if config['dummy']:

                                        # Update phase profile and retrieve S-H spot image 
                                        if i == 0:

                                            # Option 1: Generate real zernike phase profile using DM control matrix
                                            if config['real_zernike']:

                                                # Generate input zernike coefficient array
                                                self.zern_coeff[j + 2] = config['zern_test']['incre_amp'] * (k + 1)

                                                # Retrieve actuator voltages from zernike coefficient array
                                                voltages = np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                    [:,:config['AO']['control_coeff_num']], self.zern_coeff))
                                                
                                                # Generate zernike phase profile from DM
                                                phase_init = self.phase_calc(voltages)

                                                # Check whether need to incorporate sample reflectance process
                                                if config['reflect_on'] == 1:
                                                    phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam)
                                                
                                            # Option 2: Generate ideal zernike phase profile
                                            else:
                                                
                                                # Generate input zernike coefficient array
                                                self.zern_coeff[j + 2] = config['zern_test']['incre_amp'] * (k + 1)
                                                
                                                # Generate ideal zernike phase profile
                                                phase_init = zern_phase(self.SB_settings,  self.zern_coeff)

                                                # Check whether need to incorporate sample reflectance process
                                                if config['reflect_on'] == 1:
                                                    phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam) 

                                            # Display initial phase
                                            self.image.emit(phase_init)

                                            # print('\nMax and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase_init), np.amin(phase_init)))

                                            # Get simulated S-H spots and append to list
                                            AO_image, spot_cent_x, spot_cent_y = fft_spot_from_phase(self.SB_settings, phase_init)
                                            dset_append(data_set_1, 'dummy_AO_img', AO_image)
                                            dset_append(data_set_1, 'dummy_spot_cent_x', spot_cent_x)
                                            dset_append(data_set_1, 'dummy_spot_cent_y', spot_cent_y)
                                            
                                            phase = phase_init.copy()

                                        else:

                                            # Calculate phase profile introduced by DM
                                            delta_phase = self.phase_calc(voltages)

                                            # Update phase data
                                            phase = phase_init - delta_phase

                                            # Display corrected phase
                                            self.image.emit(phase)

                                            # print('Max and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase), np.amin(phase)))

                                            # Get simulated S-H spots and append to list
                                            AO_image, spot_cent_x, spot_cent_y = fft_spot_from_phase(self.SB_settings, phase)
                                            dset_append(data_set_1, 'dummy_AO_img', AO_image)
                                            dset_append(data_set_1, 'dummy_spot_cent_x', spot_cent_x)
                                            dset_append(data_set_1, 'dummy_spot_cent_y', spot_cent_y)

                                    else:

                                        # Send values vector to mirror
                                        self.mirror.Send(voltages)
                                        
                                        # Wait for DM to settle
                                        time.sleep(config['DM']['settling_time'])
                                    
                                        # Acquire S-H spots using camera
                                        AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                        AO_image = np.mean(AO_image_stack, axis = 2)

                                    # Image thresholding to remove background
                                    AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                    AO_image[AO_image < 0] = 0
                                    self.image.emit(AO_image)

                                    # Append image to list
                                    dset_append(data_set_1, 'real_AO_img', AO_image)

                                    # Calculate centroids of S-H spots
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2) 
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                    # Draw actual S-H spot centroids on image layer
                                    AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                    self.image.emit(AO_image)

                                    # Take tip\tilt off
                                    slope_x -= np.mean(slope_x)
                                    slope_y -= np.mean(slope_y)

                                    # Concatenate slopes into one slope matrix
                                    slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                    # Get detected zernike coefficients from slope matrix
                                    self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                                    # Get residual zernike error and calculate root mean square (rms) error, Strehl ratio
                                    zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                                    zern_err_part[[0, 1], 0] = 0
                                    rms_zern = np.sqrt((zern_err ** 2).sum())
                                    rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                                    strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)

                                    if i == 0:
                                        self.det_cor_zern_coeff[2 * j, :] = self.zern_coeff_detect[2:config['AO']['control_coeff_num'], 0].T
                                        self.det_cor_rms_zern[2 * j, 0] = rms_zern_part
                                        self.det_cor_strehl[2 * j, 0] = strehl

                                    print('Root mean square error {} is {} um'.format(i, rms_zern_part))
                                    print('Strehl ratio {} is {}'.format(i, strehl))

                                    # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                                    if strehl >= config['AO']['tolerance_fact_strehl'] or i == self.AO_settings['loop_max']:
                                        self.zern_coeff[j + 2] = 0
                                        self.det_cor_zern_coeff[2 * j + 1, :] = self.zern_coeff_detect[2:config['AO']['control_coeff_num'], 0].T
                                        self.det_cor_rms_zern[2 * j + 1, 0] = rms_zern_part
                                        self.det_cor_strehl[2 * j + 1, 0] = strehl
                                        break                 

                                except Exception as e:
                                    print(e)
                            else:

                                self.done.emit()

                    # Save data to file
                    sp.io.savemat('zern_gen_det_cor/zernike_correction/amp_' + str(config['zern_test']['incre_amp'] * (k + 1)) + '_zern_amp_run' + str(n) + '.mat',\
                        dict(zern_det_cor_zern_amp = self.det_cor_zern_coeff))
                    sp.io.savemat('zern_gen_det_cor/zernike_correction/amp_' + str(config['zern_test']['incre_amp'] * (k + 1)) + '_rms_zern_run' + str(n) + '.mat',\
                        dict(zern_det_cor_rms_zern = self.det_cor_rms_zern))
                    sp.io.savemat('zern_gen_det_cor/zernike_correction/amp_' + str(config['zern_test']['incre_amp'] * (k + 1)) + '_strehl_run' + str(n) + '.mat',\
                        dict(zern_det_cor_strehl = self.det_cor_strehl))

                    # Close HDF5 file
                    data_file.close()

            # Save accurate Zernike mode voltages to file
            if config['zern_test']['save_voltages']:
                sp.io.savemat('zern_volts/zern_volts_' + str(config['zern_test']['incre_num']) + '_' + str(config['zern_test']['incre_amp']) + '.mat',\
                    dict(zern_volts = self.zern_volts))

            self.message.emit('\nProcess complete.')

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is:', (prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:
                
                self.AO_info['zern_test']['zern_det_cor_zern_amp_via_zern'] = self.det_cor_zern_coeff
                self.AO_info['zern_test']['zern_det_cor_rms_zern_via_zern'] = self.det_cor_rms_zern
                self.AO_info['zern_test']['zern_det_cor_strehl_via_zern'] = self.det_cor_strehl

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit()

            # Finished closed-loop AO process
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)

    @Slot(object)
    def run1(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Run closed-loop AO correction for each generated zernike mode aberration via slopes control for all 'control_coeff_num' modes 
            and multiple amplitudes of incremental steps 
            """
            # Save calibration slope values and zernike influence function to file
            sp.io.savemat('zern_gen_det_cor/calib_slope_x.mat', dict(calib_slope_x = self.mirror_settings['calib_slope_x']))
            sp.io.savemat('zern_gen_det_cor/calib_slope_y.mat', dict(calib_slope_y = self.mirror_settings['calib_slope_y']))
            sp.io.savemat('zern_gen_det_cor/inf_matrix_zern.mat', dict(inf_matrix_zern = self.mirror_settings['inf_matrix_zern']))

            # Get number of Zernike modes to generate
            zern_num = config['AO']['control_coeff_num'] - 2

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            self.message.emit('\nProcess started for closed-loop AO via slopes...')

            prev1 = time.perf_counter()

            for n in range(config['zern_test']['run_num']):

                # Run closed-loop control for each zernike mode aberration
                for k in range(config['zern_test']['incre_num']):

                    # Initialise AO information parameter
                    self.AO_info = {'zern_test': {}}

                    # Create new datasets in HDF5 file to store closed-loop AO data and open file
                    get_dset(self.SB_settings, 'zern_test', flag = 0)
                    data_file = h5py.File('data_info.h5', 'a')
                    data_set_1 = data_file['AO_img']['zern_test']
                    data_set_2 = data_file['AO_info']['zern_test']

                    # Initialise array to store initial and final detected value of each generated Zernike mode 
                    # and RMS Zernike value / Strehl ratio considering recon_coeff_num
                    self.det_cor_zern_coeff = np.zeros([zern_num * 2, zern_num])
                    self.det_cor_rms_zern = np.zeros([zern_num * 2, 1])
                    self.det_cor_strehl = np.zeros([zern_num * 2, 1])

                    # Determine the amplitude to be generated for each Zernike mode
                    zern_amp_gen = config['zern_test']['incre_amp'] * (k + 1)

                    # Determine initial loop gain for generation of each Zernike mode
                    if zern_amp_gen <= 0.2:
                        loop_gain_gen = 0.2
                    elif zern_amp_gen > 0.2:
                        loop_gain_gen = 0.3

                    for j in range(zern_num):

                        print('On amplitude {} Zernike mode {}'.format(k + 1, j + 3))

                        for i in range(self.AO_settings['loop_max'] + 1):
                            
                            if self.loop:
                                
                                try:

                                    # Update mirror control voltages
                                    if i == 0:

                                        if not config['dummy']:

                                            # Generate one Zernike mode on DM for correction each time
                                            self.zern_coeff[j + 2, 0] = zern_amp_gen

                                            # Run closed-loop to generate a precise amount of Zernike modes using DM
                                            for m in range(config['zern_test']['loop_max_gen']):

                                                if m == 0:

                                                    voltages[:] = config['DM']['vol_bias']

                                                else:

                                                    # Update control voltages
                                                    voltages -= loop_gain_gen * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                        [:,:config['AO']['control_coeff_num']], (zern_array_det[:config['AO']['control_coeff_num']] - self.zern_coeff)))

                                                # Send values vector to mirror
                                                self.mirror.Send(voltages)

                                                # Wait for DM to settle
                                                time.sleep(config['DM']['settling_time'])
                                            
                                                # Acquire S-H spots using camera
                                                AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                                AO_image = np.mean(AO_image_stack, axis = 2)

                                                # Image thresholding to remove background
                                                AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                                AO_image[AO_image < 0] = 0
                                                self.image.emit(AO_image)

                                                # Append image to list
                                                dset_append(data_set_1, 'real_AO_img', AO_image)

                                                # Calculate centroids of S-H spots
                                                act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2)
                                                act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                                # Draw actual S-H spot centroids on image layer
                                                AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                                self.image.emit(AO_image)

                                                # Take tip\tilt off
                                                slope_x -= np.mean(slope_x)
                                                slope_y -= np.mean(slope_y)

                                                # Concatenate slopes into one slope matrix
                                                slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                                # Get detected zernike coefficients from slope matrix
                                                zern_array_det = np.dot(self.mirror_settings['conv_matrix'], slope)

                                                # print('Detected amplitude of mode {} is {} um'.format(m + 3, zern_array_det[j + 2, 0]))

                                                if abs(zern_array_det[j + 2, 0] - zern_amp_gen) / zern_amp_gen <= 0.075 or m == config['zern_test']['loop_max_gen'] - 1:
                                                    if config['zern_test']['save_voltages']:
                                                        self.zern_volts[:, j, k] = voltages
                                                    break
                                        else:

                                            voltages[:] = config['DM']['vol_bias']
                                    else:

                                        voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_slopes'], slope_err))

                                        # print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

                                    if config['dummy']:

                                        # Update phase profile and retrieve S-H spot image 
                                        if i == 0:

                                            # Option 1: Generate real zernike phase profile using DM control matrix
                                            if config['real_zernike']:

                                                # Generate input zernike coefficient array
                                                self.zern_coeff[j + 2] = config['zern_test']['incre_amp'] * (k + 1)

                                                # Retrieve actuator voltages from zernike coefficient array
                                                voltages = np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                    [:,:config['AO']['control_coeff_num']], self.zern_coeff))
                                                
                                                # Generate zernike phase profile from DM
                                                phase_init = self.phase_calc(voltages)

                                                # Check whether need to incorporate sample reflectance process
                                                if config['reflect_on'] == 1:
                                                    phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam)
                                                
                                            # Option 2: Generate ideal zernike phase profile
                                            else:
                                                
                                                # Generate input zernike coefficient array
                                                self.zern_coeff[j + 2] = config['zern_test']['incre_amp'] * (k + 1)
                                                
                                                # Generate ideal zernike phase profile
                                                phase_init = zern_phase(self.SB_settings,  self.zern_coeff)

                                                # Check whether need to incorporate sample reflectance process
                                                if config['reflect_on'] == 1:
                                                    phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam)

                                            # Display initial phase
                                            self.image.emit(phase_init)

                                            # print('\nMax and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase_init), np.amin(phase_init)))

                                            # Get simulated S-H spots and append to list
                                            AO_image, spot_cent_x, spot_cent_y = fft_spot_from_phase(self.SB_settings, phase_init)
                                            dset_append(data_set_1, 'dummy_AO_img', AO_image)
                                            dset_append(data_set_1, 'dummy_spot_cent_x', spot_cent_x)
                                            dset_append(data_set_1, 'dummy_spot_cent_y', spot_cent_y)
                                            
                                            phase = phase_init.copy()

                                        else:

                                            # Calculate phase profile introduced by DM
                                            delta_phase = self.phase_calc(voltages)

                                            # Update phase data
                                            phase = phase_init - delta_phase

                                            # Display corrected phase
                                            self.image.emit(phase)

                                            # print('Max and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase), np.amin(phase)))

                                            # Get simulated S-H spots and append to list
                                            AO_image, spot_cent_x, spot_cent_y = fft_spot_from_phase(self.SB_settings, phase)
                                            dset_append(data_set_1, 'dummy_AO_img', AO_image)
                                            dset_append(data_set_1, 'dummy_spot_cent_x', spot_cent_x)
                                            dset_append(data_set_1, 'dummy_spot_cent_y', spot_cent_y)

                                    else:

                                        # Send values vector to mirror
                                        self.mirror.Send(voltages)
                                        
                                        # Wait for DM to settle
                                        time.sleep(config['DM']['settling_time'])
                                    
                                        # Acquire S-H spots using camera
                                        AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                        AO_image = np.mean(AO_image_stack, axis = 2)

                                    # Image thresholding to remove background
                                    AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                    AO_image[AO_image < 0] = 0
                                    self.image.emit(AO_image)

                                    # Append image to list
                                    dset_append(data_set_1, 'real_AO_img', AO_image)

                                    # Calculate centroids of S-H spots
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2) 
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                    # Draw actual S-H spot centroids on image layer
                                    AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                    self.image.emit(AO_image)

                                    # Take tip\tilt off
                                    slope_x -= np.mean(slope_x)
                                    slope_y -= np.mean(slope_y)

                                    # Concatenate slopes into one slope matrix
                                    slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                    # Get residual slope error
                                    slope_err = slope.copy()

                                    # Get detected zernike coefficients from slope matrix
                                    self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                                    # Get residual zernike error and calculate root mean square (rms) error, Strehl ratio
                                    zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                                    zern_err_part[[0, 1], 0] = 0
                                    rms_zern = np.sqrt((zern_err ** 2).sum())
                                    rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                                    strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)
                                    
                                    if i == 0:
                                        self.det_cor_zern_coeff[2 * j, :] = self.zern_coeff_detect[2:config['AO']['control_coeff_num'], 0].T
                                        self.det_cor_rms_zern[2 * j, 0] = rms_zern_part
                                        self.det_cor_strehl[2 * j, 0] = strehl

                                    print('Root mean square error {} is {} um'.format(i, rms_zern_part))
                                    print('Strehl ratio {} is {}'.format(i, strehl))

                                    # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                                    if strehl >= config['AO']['tolerance_fact_strehl'] or i == self.AO_settings['loop_max']:
                                        self.zern_coeff[j + 2] = 0
                                        self.det_cor_zern_coeff[2 * j + 1, :] = self.zern_coeff_detect[2:config['AO']['control_coeff_num'], 0].T
                                        self.det_cor_rms_zern[2 * j + 1, 0] = rms_zern_part
                                        self.det_cor_strehl[2 * j + 1, 0] = strehl
                                        break                 

                                except Exception as e:
                                    print(e)
                            else:

                                self.done.emit()

                    # Save data to file
                    sp.io.savemat('zern_gen_det_cor/zernike_correction/amp_' + str(config['zern_test']['incre_amp'] * (k + 1)) + '_zern_amp_run' + str(n) + '.mat',\
                        dict(zern_det_cor_zern_amp = self.det_cor_zern_coeff))
                    sp.io.savemat('zern_gen_det_cor/zernike_correction/amp_' + str(config['zern_test']['incre_amp'] * (k + 1)) + '_rms_zern_run' + str(n) + '.mat',\
                        dict(zern_det_cor_rms_zern = self.det_cor_rms_zern))
                    sp.io.savemat('zern_gen_det_cor/zernike_correction/amp_' + str(config['zern_test']['incre_amp'] * (k + 1)) + '_strehl_run' + str(n) + '.mat',\
                        dict(zern_det_cor_strehl = self.det_cor_strehl))

                    # Close HDF5 file
                    data_file.close()

            # Save accurate Zernike mode voltages to file
            if config['zern_test']['save_voltages']:
                sp.io.savemat('zern_volts/zern_volts_' + str(config['zern_test']['incre_num']) + '_' + str(config['zern_test']['incre_amp']) + '.mat',\
                    dict(zern_volts = self.zern_volts))

            self.message.emit('\nProcess complete.')

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is:', (prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:
                
                self.AO_info['zern_test']['zern_det_cor_zern_amp_via_slopes'] = self.det_cor_zern_coeff
                self.AO_info['zern_test']['zern_det_cor_rms_zern_via_slopes'] = self.det_cor_rms_zern
                self.AO_info['zern_test']['zern_det_cor_strehl_via_slopes'] = self.det_cor_strehl

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit()

            # Finished closed-loop AO process
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)

    @Slot(object)
    def run2(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Run closed-loop AO correction for some specific zernike mode aberrations via Zernike control with a fixed amplitude 
            """
            # Save calibration slope values and zernike influence function to file
            # sp.io.savemat('zern_gen_det_cor_full/calib_slope_x.mat', dict(calib_slope_x = self.mirror_settings['calib_slope_x']))
            # sp.io.savemat('zern_gen_det_cor_full/calib_slope_y.mat', dict(calib_slope_y = self.mirror_settings['calib_slope_y']))
            # sp.io.savemat('zern_gen_det_cor_full/inf_matrix_zern.mat', dict(inf_matrix_zern = self.mirror_settings['inf_matrix_zern']))
            sp.io.savemat('zern_gen_det_cor_mult_full/calib_slope_x.mat', dict(calib_slope_x = self.mirror_settings['calib_slope_x']))
            sp.io.savemat('zern_gen_det_cor_mult_full/calib_slope_y.mat', dict(calib_slope_y = self.mirror_settings['calib_slope_y']))
            sp.io.savemat('zern_gen_det_cor_mult_full/inf_matrix_zern.mat', dict(inf_matrix_zern = self.mirror_settings['inf_matrix_zern']))

            # Initialise AO information parameter
            self.AO_info = {'zern_test': {}}

            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'zern_test', flag = 0)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['zern_test']
            data_set_2 = data_file['AO_info']['zern_test']

            # Initialise zernike mode array, zernike amplitude array, and loop_gain_gen array
            # zern_mode_array = [4, 6, 11, 19]
            # zern_amp_array = [0.3, 0.15, 0.15, 0.2]
            # loop_gain_gen_array = [0.3, 0.2, 0.2, 0.2]

            # Get number of Zernike modes to generate
            # zern_num = len(zern_mode_array)

            # For multiple modes 1
            zern_mode_array = [4, 6, 11, 19]
            zern_array_temp = [0, 0, 0, 0, 0.1, 0, 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.1]
            zern_num = 1

            # For multiple modes 2
            # zern_mode_array = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            # zern_array_temp = [0, 0, 0.05, 0.03, 0.05, 0.02, 0.08, 0.02, 0.01, 0.04, 0.01, 0.06, 0.02, 0.02, 0.01, 0.015, 0.01, 0.01, 0.02, 0.01]
            # zern_num = 1

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            self.message.emit('\nProcess started for closed-loop AO via Zernikes...')

            prev1 = time.perf_counter()

            for n in range(config['zern_test']['run_num']):

                # Run closed-loop control for each specified zernike mode
                for j in range(zern_num):

                    # print('On Zernike mode {}'.format(zern_mode_array[j] + 1))

                    # Initialise array to store detected values of each generated Zernike mode and RMS Zernike value / Strehl ratio considering recon_coeff_num
                    # for each iteration
                    self.det_cor_zern_coeff = np.zeros([self.AO_settings['loop_max'] + 1, config['AO']['control_coeff_num'] - 2])
                    self.det_cor_rms_zern = np.zeros([self.AO_settings['loop_max'] + 1, 1])
                    self.det_cor_strehl = np.zeros([self.AO_settings['loop_max'] + 1, 1])

                    # Get initial loop_gain_gen
                    # loop_gain_gen = loop_gain_gen_array[j]
                    loop_gain_gen = 0.2

                    for i in range(self.AO_settings['loop_max'] + 1):
                        
                        if self.loop:
                            
                            try:

                                # Update mirror control voltages
                                if i == 0:

                                    if not config['dummy']:

                                        # Generate one Zernike mode on DM for correction each time
                                        # self.zern_coeff[zern_mode_array[j], 0] = zern_amp_array[j]
                                        self.zern_coeff[:len(zern_array_temp), 0] = zern_array_temp

                                        # Run closed-loop to generate a precise amount of Zernike modes using DM
                                        for m in range(config['zern_test']['loop_max_gen']):

                                            if m == 0:

                                                voltages[:] = config['DM']['vol_bias']

                                            else:

                                                # Update control voltages
                                                voltages -= loop_gain_gen * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                    [:,:config['AO']['control_coeff_num']], (zern_array_det[:config['AO']['control_coeff_num']] - self.zern_coeff)))

                                            # Send values vector to mirror
                                            self.mirror.Send(voltages)

                                            # Wait for DM to settle
                                            time.sleep(config['DM']['settling_time'])
                                        
                                            # Acquire S-H spots using camera
                                            AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                            AO_image = np.mean(AO_image_stack, axis = 2)

                                            # Image thresholding to remove background
                                            AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                            AO_image[AO_image < 0] = 0
                                            self.image.emit(AO_image)

                                            # Append image to list
                                            dset_append(data_set_1, 'real_AO_img', AO_image)

                                            # Calculate centroids of S-H spots
                                            act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2)
                                            act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                            # Draw actual S-H spot centroids on image layer
                                            AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                            self.image.emit(AO_image)

                                            # Take tip\tilt off
                                            slope_x -= np.mean(slope_x)
                                            slope_y -= np.mean(slope_y)

                                            # Concatenate slopes into one slope matrix
                                            slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                            # Get detected zernike coefficients from slope matrix
                                            zern_array_det = np.dot(self.mirror_settings['conv_matrix'], slope)

                                            # print('Detected amplitude of mode {} is {} um'.format(zern_mode_array[j] + 1, zern_array_det[zern_mode_array[j], 0]))
                                            print('Detected amplitude of mode {} is {} um'.format(zern_mode_array[0] + 1, zern_array_det[zern_mode_array[0], 0]))
                                            print('Detected amplitude of mode {} is {} um'.format(zern_mode_array[1] + 1, zern_array_det[zern_mode_array[1], 0]))
                                            print('Detected amplitude of mode {} is {} um'.format(zern_mode_array[2] + 1, zern_array_det[zern_mode_array[2], 0]))
                                            print('Detected amplitude of mode {} is {} um'.format(zern_mode_array[3] + 1, zern_array_det[zern_mode_array[3], 0]))
                                            # print('Detected amplitude of mode {} is {} um'.format(zern_mode_array[4] + 1, zern_array_det[zern_mode_array[4], 0]))

                                            # if abs(zern_array_det[zern_mode_array[j], 0] - zern_amp_array[j]) / zern_amp_array[j] <= 0.075:
                                            #     break
                                    else:

                                        voltages[:] = config['DM']['vol_bias']                              
                                else:

                                    voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                        [:,:config['AO']['control_coeff_num']], zern_err[:config['AO']['control_coeff_num']]))

                                    print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

                                if config['dummy']:

                                    # Update phase profile and retrieve S-H spot image 
                                    if i == 0:

                                        # Option 1: Generate real zernike phase profile using DM control matrix
                                        if config['real_zernike']:

                                            # Generate input zernike coefficient array
                                            self.zern_coeff[j + 2] = config['zern_test']['incre_amp'] * (k + 1)

                                            # Retrieve actuator voltages from zernike coefficient array
                                            voltages = np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                [:,:config['AO']['control_coeff_num']], self.zern_coeff))
                                            
                                            # Generate zernike phase profile from DM
                                            phase_init = self.phase_calc(voltages)

                                            # Check whether need to incorporate sample reflectance process
                                            if config['reflect_on'] == 1:
                                                phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam)
                                            
                                        # Option 2: Generate ideal zernike phase profile
                                        else:
                                            
                                            # Generate input zernike coefficient array
                                            self.zern_coeff[j + 2] = config['zern_test']['incre_amp'] * (k + 1)
                                            
                                            # Generate ideal zernike phase profile
                                            phase_init = zern_phase(self.SB_settings,  self.zern_coeff)

                                            # Check whether need to incorporate sample reflectance process
                                            if config['reflect_on'] == 1:
                                                phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam) 

                                        # Display initial phase
                                        self.image.emit(phase_init)

                                        # print('\nMax and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase_init), np.amin(phase_init)))

                                        # Get simulated S-H spots and append to list
                                        AO_image, spot_cent_x, spot_cent_y = fft_spot_from_phase(self.SB_settings, phase_init)
                                        dset_append(data_set_1, 'dummy_AO_img', AO_image)
                                        dset_append(data_set_1, 'dummy_spot_cent_x', spot_cent_x)
                                        dset_append(data_set_1, 'dummy_spot_cent_y', spot_cent_y)
                                        
                                        phase = phase_init.copy()

                                    else:

                                        # Calculate phase profile introduced by DM
                                        delta_phase = self.phase_calc(voltages)

                                        # Update phase data
                                        phase = phase_init - delta_phase

                                        # Display corrected phase
                                        self.image.emit(phase)

                                        # print('Max and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase), np.amin(phase)))

                                        # Get simulated S-H spots and append to list
                                        AO_image, spot_cent_x, spot_cent_y = fft_spot_from_phase(self.SB_settings, phase)
                                        dset_append(data_set_1, 'dummy_AO_img', AO_image)
                                        dset_append(data_set_1, 'dummy_spot_cent_x', spot_cent_x)
                                        dset_append(data_set_1, 'dummy_spot_cent_y', spot_cent_y)

                                else:

                                    # Send values vector to mirror
                                    self.mirror.Send(voltages)
                                    
                                    # Wait for DM to settle
                                    time.sleep(config['DM']['settling_time'])
                                
                                    # Acquire S-H spots using camera
                                    AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                    AO_image = np.mean(AO_image_stack, axis = 2)

                                # Image thresholding to remove background
                                AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                AO_image[AO_image < 0] = 0
                                self.image.emit(AO_image)

                                # Append image to list
                                dset_append(data_set_1, 'real_AO_img', AO_image)

                                if i == 0:
                                    imsave('zern_gen_det_cor_mult_full/zernike_correction/SH_spots_before_run' + str(n) + '.tif',\
                                        AO_image.astype(np.float32))

                                # Calculate centroids of S-H spots
                                act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2) 
                                act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                # Draw actual S-H spot centroids on image layer
                                AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                self.image.emit(AO_image)

                                # Take tip\tilt off
                                slope_x -= np.mean(slope_x)
                                slope_y -= np.mean(slope_y)

                                if i == 0:
                                    # sp.io.savemat('zern_gen_det_cor_full/zernike_correction/mode' + str(zern_mode_array[j] + 1) + '_slope_x_before_run' + str(n) + '.mat',\
                                    #     dict(zern_det_cor_slope_x_before = slope_x))
                                    # sp.io.savemat('zern_gen_det_cor_full/zernike_correction/mode' + str(zern_mode_array[j] + 1) + '_slope_y_before_run' + str(n) + '.mat',\
                                    #     dict(zern_det_cor_slope_y_before = slope_y))
                                    sp.io.savemat('zern_gen_det_cor_mult_full/zernike_correction/slope_x_before_run' + str(n) + '.mat',\
                                        dict(zern_det_cor_slope_x_before = slope_x))
                                    sp.io.savemat('zern_gen_det_cor_mult_full/zernike_correction/slope_y_before_run' + str(n) + '.mat',\
                                        dict(zern_det_cor_slope_y_before = slope_y))

                                # Concatenate slopes into one slope matrix
                                slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                # Get detected zernike coefficients from slope matrix
                                self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                                # Get residual zernike error and calculate root mean square (rms) error, Strehl ratio
                                zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                                zern_err_part[[0, 1], 0] = 0
                                rms_zern = np.sqrt((zern_err ** 2).sum())
                                rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                                strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)

                                self.det_cor_zern_coeff[i, :] = self.zern_coeff_detect[2:config['AO']['control_coeff_num'], 0].T
                                self.det_cor_rms_zern[i, 0] = rms_zern_part
                                self.det_cor_strehl[i, 0] = strehl

                                print('Root mean square error {} is {} um'.format(i, rms_zern_part))
                                print('Strehl ratio {} is {}'.format(i, strehl))

                                # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                                if strehl >= config['AO']['tolerance_fact_strehl'] or i == self.AO_settings['loop_max']:
                                    # self.zern_coeff[zern_mode_array[j], 0] = 0
                                    # sp.io.savemat('zern_gen_det_cor_full/zernike_correction/mode' + str(zern_mode_array[j] + 1) + '_slope_x_after_run' + str(n) + '.mat',\
                                    #     dict(zern_det_cor_slope_x_after = slope_x))
                                    # sp.io.savemat('zern_gen_det_cor_full/zernike_correction/mode' + str(zern_mode_array[j] + 1) + '_slope_y_after_run' + str(n) + '.mat',\
                                    #     dict(zern_det_cor_slope_y_after = slope_y))
                                    # imsave('zern_gen_det_cor_full/zernike_correction/mode' + str(zern_mode_array[j] + 1) + '_SH_spots_after_run' + str(n) + '.tif',\
                                    #     AO_image.astype(np.float32))
                                    self.zern_coeff[:, 0] = 0
                                    sp.io.savemat('zern_gen_det_cor_mult_full/zernike_correction/slope_x_after_run' + str(n) + '.mat',\
                                        dict(zern_det_cor_slope_x_after = slope_x))
                                    sp.io.savemat('zern_gen_det_cor_mult_full/zernike_correction/slope_y_after_run' + str(n) + '.mat',\
                                        dict(zern_det_cor_slope_y_after = slope_y))
                                    imsave('zern_gen_det_cor_mult_full/zernike_correction/SH_spots_after_run' + str(n) + '.tif',\
                                        AO_image.astype(np.float32))
                                    break                 

                            except Exception as e:
                                print(e)
                        else:

                            self.done.emit()

                    # Save data to file
                    # sp.io.savemat('zern_gen_det_cor_full/zernike_correction/mode' + str(zern_mode_array[j] + 1) + '_zern_amp_run' + str(n) + '.mat',\
                    #     dict(zern_det_cor_zern_amp = self.det_cor_zern_coeff))
                    # sp.io.savemat('zern_gen_det_cor_full/zernike_correction/mode' + str(zern_mode_array[j] + 1) + '_rms_zern_run' + str(n) + '.mat',\
                    #     dict(zern_det_cor_rms_zern = self.det_cor_rms_zern))
                    # sp.io.savemat('zern_gen_det_cor_full/zernike_correction/mode' + str(zern_mode_array[j] + 1) + '_strehl_run' + str(n) + '.mat',\
                    #     dict(zern_det_cor_strehl = self.det_cor_strehl))
                    sp.io.savemat('zern_gen_det_cor_mult_full/zernike_correction/zern_amp_run' + str(n) + '.mat',\
                        dict(zern_det_cor_zern_amp = self.det_cor_zern_coeff))
                    sp.io.savemat('zern_gen_det_cor_mult_full/zernike_correction/rms_zern_run' + str(n) + '.mat',\
                        dict(zern_det_cor_rms_zern = self.det_cor_rms_zern))
                    sp.io.savemat('zern_gen_det_cor_mult_full/zernike_correction/strehl_run' + str(n) + '.mat',\
                        dict(zern_det_cor_strehl = self.det_cor_strehl))

            # Close HDF5 file
            data_file.close()

            self.message.emit('\nProcess complete.')

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is:', (prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:
                
                self.AO_info['zern_test']['zern_det_cor_zern_amp_via_zern'] = self.det_cor_zern_coeff
                self.AO_info['zern_test']['zern_det_cor_rms_zern_via_zern'] = self.det_cor_rms_zern
                self.AO_info['zern_test']['zern_det_cor_strehl_via_zern'] = self.det_cor_strehl

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit()

            # Finished closed-loop AO process
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)

    @Slot(object)
    def run3(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Run closed-loop AO correction for some specific zernike mode aberrations via slopes control with a fixed amplitude 
            """
            # Save calibration slope values and zernike influence function to file
            # sp.io.savemat('zern_gen_det_cor_full/calib_slope_x.mat', dict(calib_slope_x = self.mirror_settings['calib_slope_x']))
            # sp.io.savemat('zern_gen_det_cor_full/calib_slope_y.mat', dict(calib_slope_y = self.mirror_settings['calib_slope_y']))
            # sp.io.savemat('zern_gen_det_cor_full/inf_matrix_zern.mat', dict(inf_matrix_zern = self.mirror_settings['inf_matrix_zern']))
            sp.io.savemat('zern_gen_det_cor_mult_full/calib_slope_x.mat', dict(calib_slope_x = self.mirror_settings['calib_slope_x']))
            sp.io.savemat('zern_gen_det_cor_mult_full/calib_slope_y.mat', dict(calib_slope_y = self.mirror_settings['calib_slope_y']))
            sp.io.savemat('zern_gen_det_cor_mult_full/inf_matrix_zern.mat', dict(inf_matrix_zern = self.mirror_settings['inf_matrix_zern']))

            # Initialise AO information parameter
            self.AO_info = {'zern_test': {}}

            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'zern_test', flag = 0)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['zern_test']
            data_set_2 = data_file['AO_info']['zern_test']

            # Initialise zernike mode array, zernike amplitude array, and loop_gain_gen array
            # zern_mode_array = [4, 6, 11, 19]
            # zern_amp_array = [0.3, 0.15, 0.15, 0.2]
            # loop_gain_gen_array = [0.3, 0.2, 0.2, 0.2]

            # Get number of Zernike modes to generate
            # zern_num = len(zern_mode_array)

            # For multiple modes 1
            zern_mode_array = [4, 6, 11, 19]
            zern_array_temp = [0, 0, 0, 0, 0.1, 0, 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.1]
            zern_num = 1
            
            # For multiple modes 2
            # zern_mode_array = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            # zern_array_temp = [0, 0, 0.05, 0.03, 0.05, 0.02, 0.08, 0.02, 0.01, 0.04, 0.01, 0.06, 0.02, 0.02, 0.01, 0.015, 0.01, 0.01, 0.02, 0.01]
            # zern_num = 1

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            self.message.emit('\nProcess started for closed-loop AO via slopes...')

            prev1 = time.perf_counter()

            for n in range(config['zern_test']['run_num']):

                # Run closed-loop control for each specified zernike mode
                for j in range(zern_num):

                    # print('On Zernike mode {}'.format(zern_mode_array[j] + 1))

                    # Initialise array to store detected values of each generated Zernike mode and RMS Zernike value / Strehl ratio considering recon_coeff_num
                    # for each iteration
                    self.det_cor_zern_coeff = np.zeros([self.AO_settings['loop_max'] + 1, config['AO']['control_coeff_num'] - 2])
                    self.det_cor_rms_zern = np.zeros([self.AO_settings['loop_max'] + 1, 1])
                    self.det_cor_strehl = np.zeros([self.AO_settings['loop_max'] + 1, 1])

                    # Get initial loop_gain_gen
                    # loop_gain_gen = loop_gain_gen_array[j]
                    loop_gain_gen = 0.2

                    for i in range(self.AO_settings['loop_max'] + 1):
                        
                        if self.loop:
                            
                            try:

                                # Update mirror control voltages
                                if i == 0:

                                    if not config['dummy']:

                                        # Generate one Zernike mode on DM for correction each time
                                        # self.zern_coeff[zern_mode_array[j], 0] = zern_amp_array[j]
                                        self.zern_coeff[:len(zern_array_temp), 0] = zern_array_temp

                                        # Run closed-loop to generate a precise amount of Zernike modes using DM
                                        for m in range(config['zern_test']['loop_max_gen']):

                                            if m == 0:

                                                voltages[:] = config['DM']['vol_bias']

                                            else:

                                                # Update control voltages
                                                voltages -= loop_gain_gen * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                    [:,:config['AO']['control_coeff_num']], (zern_array_det[:config['AO']['control_coeff_num']] - self.zern_coeff)))

                                            # Send values vector to mirror
                                            self.mirror.Send(voltages)

                                            # Wait for DM to settle
                                            time.sleep(config['DM']['settling_time'])
                                        
                                            # Acquire S-H spots using camera
                                            AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                            AO_image = np.mean(AO_image_stack, axis = 2)

                                            # Image thresholding to remove background
                                            AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                            AO_image[AO_image < 0] = 0
                                            self.image.emit(AO_image)

                                            # Append image to list
                                            dset_append(data_set_1, 'real_AO_img', AO_image)

                                            # Calculate centroids of S-H spots
                                            act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2)
                                            act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                            # Draw actual S-H spot centroids on image layer
                                            AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                            self.image.emit(AO_image)

                                            # Take tip\tilt off
                                            slope_x -= np.mean(slope_x)
                                            slope_y -= np.mean(slope_y)

                                            # Concatenate slopes into one slope matrix
                                            slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                            # Get detected zernike coefficients from slope matrix
                                            zern_array_det = np.dot(self.mirror_settings['conv_matrix'], slope)

                                            # print('Detected amplitude of mode {} is {} um'.format(zern_mode_array[j] + 1, zern_array_det[zern_mode_array[j], 0]))
                                            print('Detected amplitude of mode {} is {} um'.format(zern_mode_array[0] + 1, zern_array_det[zern_mode_array[0], 0]))
                                            print('Detected amplitude of mode {} is {} um'.format(zern_mode_array[1] + 1, zern_array_det[zern_mode_array[1], 0]))
                                            print('Detected amplitude of mode {} is {} um'.format(zern_mode_array[2] + 1, zern_array_det[zern_mode_array[2], 0]))
                                            print('Detected amplitude of mode {} is {} um'.format(zern_mode_array[3] + 1, zern_array_det[zern_mode_array[3], 0]))
                                            # print('Detected amplitude of mode {} is {} um'.format(zern_mode_array[4] + 1, zern_array_det[zern_mode_array[4], 0]))

                                            # if abs(zern_array_det[zern_mode_array[j], 0] - zern_amp_array[j]) / zern_amp_array[j] <= 0.075:
                                            #     break
                                    else:

                                        voltages[:] = config['DM']['vol_bias']                              
                                else:

                                    voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_slopes'], slope_err))

                                    print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

                                if config['dummy']:

                                    # Update phase profile and retrieve S-H spot image 
                                    if i == 0:

                                        # Option 1: Generate real zernike phase profile using DM control matrix
                                        if config['real_zernike']:

                                            # Generate input zernike coefficient array
                                            self.zern_coeff[j + 2] = config['zern_test']['incre_amp'] * (k + 1)

                                            # Retrieve actuator voltages from zernike coefficient array
                                            voltages = np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                [:,:config['AO']['control_coeff_num']], self.zern_coeff))
                                            
                                            # Generate zernike phase profile from DM
                                            phase_init = self.phase_calc(voltages)

                                            # Check whether need to incorporate sample reflectance process
                                            if config['reflect_on'] == 1:
                                                phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam)
                                            
                                        # Option 2: Generate ideal zernike phase profile
                                        else:
                                            
                                            # Generate input zernike coefficient array
                                            self.zern_coeff[j + 2] = config['zern_test']['incre_amp'] * (k + 1)
                                            
                                            # Generate ideal zernike phase profile
                                            phase_init = zern_phase(self.SB_settings,  self.zern_coeff)

                                            # Check whether need to incorporate sample reflectance process
                                            if config['reflect_on'] == 1:
                                                phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam) 

                                        # Display initial phase
                                        self.image.emit(phase_init)

                                        # print('\nMax and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase_init), np.amin(phase_init)))

                                        # Get simulated S-H spots and append to list
                                        AO_image, spot_cent_x, spot_cent_y = fft_spot_from_phase(self.SB_settings, phase_init)
                                        dset_append(data_set_1, 'dummy_AO_img', AO_image)
                                        dset_append(data_set_1, 'dummy_spot_cent_x', spot_cent_x)
                                        dset_append(data_set_1, 'dummy_spot_cent_y', spot_cent_y)
                                        
                                        phase = phase_init.copy()

                                    else:

                                        # Calculate phase profile introduced by DM
                                        delta_phase = self.phase_calc(voltages)

                                        # Update phase data
                                        phase = phase_init - delta_phase

                                        # Display corrected phase
                                        self.image.emit(phase)

                                        # print('Max and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase), np.amin(phase)))

                                        # Get simulated S-H spots and append to list
                                        AO_image, spot_cent_x, spot_cent_y = fft_spot_from_phase(self.SB_settings, phase)
                                        dset_append(data_set_1, 'dummy_AO_img', AO_image)
                                        dset_append(data_set_1, 'dummy_spot_cent_x', spot_cent_x)
                                        dset_append(data_set_1, 'dummy_spot_cent_y', spot_cent_y)

                                else:

                                    # Send values vector to mirror
                                    self.mirror.Send(voltages)
                                    
                                    # Wait for DM to settle
                                    time.sleep(config['DM']['settling_time'])
                                
                                    # Acquire S-H spots using camera
                                    AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                    AO_image = np.mean(AO_image_stack, axis = 2)

                                # Image thresholding to remove background
                                AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                AO_image[AO_image < 0] = 0
                                self.image.emit(AO_image)

                                # Append image to list
                                dset_append(data_set_1, 'real_AO_img', AO_image)

                                if i == 0:
                                    # imsave('zern_gen_det_cor_full/slope_correction/SH_spots_before_run' + str(n) + '.tif',\
                                    #     AO_image.astype(np.float32))
                                    imsave('zern_gen_det_cor_mult_full/slope_correction/SH_spots_before_run' + str(n) + '.tif',\
                                        AO_image.astype(np.float32))

                                # Calculate centroids of S-H spots
                                act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2) 
                                act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                # Draw actual S-H spot centroids on image layer
                                AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                self.image.emit(AO_image)

                                # Take tip\tilt off
                                slope_x -= np.mean(slope_x)
                                slope_y -= np.mean(slope_y)

                                if i == 0:
                                    # sp.io.savemat('zern_gen_det_cor_full/slope_correction/mode' + str(zern_mode_array[j] + 1) + '_slope_x_before_run' + str(n) + '.mat',\
                                    #     dict(zern_det_cor_slope_x_before = slope_x))
                                    # sp.io.savemat('zern_gen_det_cor_full/slope_correction/mode' + str(zern_mode_array[j] + 1) + '_slope_y_before_run' + str(n) + '.mat',\
                                    #     dict(zern_det_cor_slope_y_before = slope_y))
                                    sp.io.savemat('zern_gen_det_cor_mult_full/slope_correction/slope_x_before_run' + str(n) + '.mat',\
                                        dict(zern_det_cor_slope_x_before = slope_x))
                                    sp.io.savemat('zern_gen_det_cor_mult_full/slope_correction/slope_y_before_run' + str(n) + '.mat',\
                                        dict(zern_det_cor_slope_y_before = slope_y))

                                # Concatenate slopes into one slope matrix
                                slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                # Get residual slope error
                                slope_err = slope.copy()
                                
                                # Get detected zernike coefficients from slope matrix
                                self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                                # Get residual zernike error and calculate root mean square (rms) error, Strehl ratio
                                zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                                zern_err_part[[0, 1], 0] = 0
                                rms_zern = np.sqrt((zern_err ** 2).sum())
                                rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                                strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)

                                self.det_cor_zern_coeff[i, :] = self.zern_coeff_detect[2:config['AO']['control_coeff_num'], 0].T
                                self.det_cor_rms_zern[i, 0] = rms_zern_part
                                self.det_cor_strehl[i, 0] = strehl

                                print('Root mean square error {} is {} um'.format(i, rms_zern_part))
                                print('Strehl ratio {} is {}'.format(i, strehl))

                                # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                                if strehl >= config['AO']['tolerance_fact_strehl'] or i == self.AO_settings['loop_max']:
                                    # self.zern_coeff[zern_mode_array[j], 0] = 0
                                    # sp.io.savemat('zern_gen_det_cor_full/slope_correction/mode' + str(zern_mode_array[j] + 1) + '_slope_x_after_run' + str(n) + '.mat',\
                                    #     dict(zern_det_cor_slope_x_after = slope_x))
                                    # sp.io.savemat('zern_gen_det_cor_full/slope_correction/mode' + str(zern_mode_array[j] + 1) + '_slope_y_after_run' + str(n) + '.mat',\
                                    #     dict(zern_det_cor_slope_y_after = slope_y))
                                    # imsave('zern_gen_det_cor_full/slope_correction/mode' + str(zern_mode_array[j] + 1) + '_SH_spots_after_run' + str(n) + '.tif',\
                                    #     AO_image.astype(np.float32))
                                    self.zern_coeff[:, 0] = 0
                                    sp.io.savemat('zern_gen_det_cor_mult_full/slope_correction/slope_x_after_run' + str(n) + '.mat',\
                                        dict(zern_det_cor_slope_x_after = slope_x))
                                    sp.io.savemat('zern_gen_det_cor_mult_full/slope_correction/slope_y_after_run' + str(n) + '.mat',\
                                        dict(zern_det_cor_slope_y_after = slope_y))
                                    imsave('zern_gen_det_cor_mult_full/slope_correction/SH_spots_after_run' + str(n) + '.tif',\
                                        AO_image.astype(np.float32))
                                    break                 

                            except Exception as e:
                                print(e)
                        else:

                            self.done.emit()

                    # Save data to file
                    # sp.io.savemat('zern_gen_det_cor_full/slope_correction/mode' + str(zern_mode_array[j] + 1) + '_zern_amp_run' + str(n) + '.mat',\
                    #     dict(zern_det_cor_zern_amp = self.det_cor_zern_coeff))
                    # sp.io.savemat('zern_gen_det_cor_full/slope_correction/mode' + str(zern_mode_array[j] + 1) + '_rms_zern_run' + str(n) + '.mat',\
                    #     dict(zern_det_cor_rms_zern = self.det_cor_rms_zern))
                    # sp.io.savemat('zern_gen_det_cor_full/slope_correction/mode' + str(zern_mode_array[j] + 1) + '_strehl_run' + str(n) + '.mat',\
                    #     dict(zern_det_cor_strehl = self.det_cor_strehl))
                    sp.io.savemat('zern_gen_det_cor_mult_full/slope_correction/zern_amp_run' + str(n) + '.mat',\
                        dict(zern_det_cor_zern_amp = self.det_cor_zern_coeff))
                    sp.io.savemat('zern_gen_det_cor_mult_full/slope_correction/rms_zern_run' + str(n) + '.mat',\
                        dict(zern_det_cor_rms_zern = self.det_cor_rms_zern))
                    sp.io.savemat('zern_gen_det_cor_mult_full/slope_correction/strehl_run' + str(n) + '.mat',\
                        dict(zern_det_cor_strehl = self.det_cor_strehl))

            # Close HDF5 file
            data_file.close()

            self.message.emit('\nProcess complete.')

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is:', (prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:
                
                self.AO_info['zern_test']['zern_det_cor_zern_amp_via_slopes'] = self.det_cor_zern_coeff
                self.AO_info['zern_test']['zern_det_cor_rms_zern_via_slopes'] = self.det_cor_rms_zern
                self.AO_info['zern_test']['zern_det_cor_strehl_via_slopes'] = self.det_cor_strehl

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit()

            # Finished closed-loop AO process
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)

    @Slot(object)
    def run4(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Perform a number of line scans across specimen and retrieve Zernike coefficients from each scan point
            """
            # Save calibration slope values and zernike influence function to file
            sp.io.savemat('xy_scan_aberr_meas/calib_slope_x.mat', dict(calib_slope_x = self.mirror_settings['calib_slope_x']))
            sp.io.savemat('xy_scan_aberr_meas/calib_slope_y.mat', dict(calib_slope_y = self.mirror_settings['calib_slope_y']))
            sp.io.savemat('xy_scan_aberr_meas/inf_matrix_zern.mat', dict(inf_matrix_zern = self.mirror_settings['inf_matrix_zern']))

            self.message.emit('\nProcess started for Zernike coefficient retrieval from different points along line scan...')

            prev1 = time.perf_counter()

            # Scan multiple points across a line in both x and y directions for a given number of loops over a large FOV (600um)
            if config['zern_test']['large_flag']:

                # Initialise array for storing retrieved zernike coefficients
                self.zern_x = np.zeros([config['AO']['recon_coeff_num'], config['zern_test']['scan_num_x_large'], config['zern_test']['loop_num']])
                self.zern_y = np.zeros([config['AO']['recon_coeff_num'], config['zern_test']['scan_num_y_large'], config['zern_test']['loop_num']])

                # Initialise array for storing retrieved slope values
                self.slope_x = np.zeros([self.SB_settings['act_ref_cent_num'] * 2, config['zern_test']['scan_num_x_large'], config['zern_test']['loop_num']])
                self.slope_y = np.zeros([self.SB_settings['act_ref_cent_num'] * 2, config['zern_test']['scan_num_y_large'], config['zern_test']['loop_num']])

                # Generate relevant amounts of x/y scan voltages (normalised) for scanning across sample
                x_array_large = np.linspace(-config['zern_test']['x_amp_large'], config['zern_test']['x_amp_large'], config['zern_test']['scan_num_x_large'])
                y_array_large = np.linspace(-config['zern_test']['y_amp_large'], config['zern_test']['y_amp_large'], config['zern_test']['scan_num_y_large'])

                for n in range(config['zern_test']['run_num']):

                    # Initialise AO information parameter
                    self.AO_info = {'zern_test': {}}

                    # Create new datasets in HDF5 file to store closed-loop AO data and open file
                    get_dset(self.SB_settings, 'zern_test', flag = 0)
                    data_file = h5py.File('data_info.h5', 'a')
                    data_set_1 = data_file['AO_img']['zern_test']
                    data_set_2 = data_file['AO_info']['zern_test']
                    
                    # Reset deformable mirror
                    self.mirror.Reset()

                    for l in range(config['zern_test']['loop_num']):

                        # Reset scanner
                        self.scanner.ResetDevicePosition()

                        print('Large FOV run {} x-scan loop {}'.format(n + 1, l + 1))

                        for m in range(config['zern_test']['scan_num_x_large']):
                        
                            if self.loop:
                                
                                try:
                                    
                                    # Send voltages to scanner
                                    self.scanner.GoToDevicePosition(x_array_large[m], 0, 255, 10)
                                
                                    # Acquire S-H spots using camera
                                    AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                    AO_image = np.mean(AO_image_stack, axis = 2)

                                    # Image thresholding to remove background
                                    AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                    AO_image[AO_image < 0] = 0
                                    self.image.emit(AO_image)

                                    # Append image to list
                                    dset_append(data_set_1, 'real_AO_img', AO_image)

                                    # Calculate centroids of S-H spots
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2) 
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                    # Draw actual S-H spot centroids on image layer
                                    AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                    self.image.emit(AO_image)

                                    # Take tip\tilt off
                                    slope_x -= np.mean(slope_x)
                                    slope_y -= np.mean(slope_y)

                                    # Concatenate slopes into one slope matrix
                                    slope = (np.concatenate((slope_x, slope_y), axis = 1)).T
                                    self.slope_x[:,m,l] = slope[:,0]

                                    # Get detected zernike coefficients from slope matrix
                                    self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                                    # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                                    zern_err = self.zern_coeff_detect.copy()
                                    rms_zern = np.sqrt((zern_err ** 2).sum())
                                    self.zern_x[:,m,l] = zern_err[:,0]

                                    print('Full zernike root mean square error {} is {} um'.format(m + 1, rms_zern))                              

                                except Exception as e:
                                    print(e)
                            else:

                                self.done.emit()

                    for l in range(config['zern_test']['loop_num']):

                        # Reset scanner
                        self.scanner.ResetDevicePosition()

                        print('Large FOV run {} y-scan loop {}'.format(n + 1, l + 1))

                        for m in range(config['zern_test']['scan_num_y_large']):
                        
                            if self.loop:
                                
                                try:

                                    # Send voltages to scanner
                                    self.scanner.GoToDevicePosition(0, y_array_large[m], 255, 10)
                                
                                    # Acquire S-H spots using camera
                                    AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                    AO_image = np.mean(AO_image_stack, axis = 2)

                                    # Image thresholding to remove background
                                    AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                    AO_image[AO_image < 0] = 0
                                    self.image.emit(AO_image)

                                    # Append image to list
                                    dset_append(data_set_1, 'real_AO_img', AO_image)

                                    # Calculate centroids of S-H spots
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2) 
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                    # Draw actual S-H spot centroids on image layer
                                    AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                    self.image.emit(AO_image)

                                    # Take tip\tilt off
                                    slope_x -= np.mean(slope_x)
                                    slope_y -= np.mean(slope_y)

                                    # Concatenate slopes into one slope matrix
                                    slope = (np.concatenate((slope_x, slope_y), axis = 1)).T
                                    self.slope_y[:,m,l] = slope[:,0]

                                    # Get detected zernike coefficients from slope matrix
                                    self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                                    # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                                    zern_err = self.zern_coeff_detect.copy()
                                    rms_zern = np.sqrt((zern_err ** 2).sum())
                                    self.zern_y[:,m,l] = zern_err[:,0]

                                    print('Full zernike root mean square error {} is {} um'.format(m + 1, rms_zern))                              

                                except Exception as e:
                                    print(e)
                            else:

                                self.done.emit()

                    sp.io.savemat('xy_scan_aberr_meas/600um/x_scan_zern_coeff_' + str(n) + '.mat', dict(x_scan_zern_coeff = self.zern_x))
                    sp.io.savemat('xy_scan_aberr_meas/600um/y_scan_zern_coeff_' + str(n) + '.mat', dict(y_scan_zern_coeff = self.zern_y))
                    sp.io.savemat('xy_scan_aberr_meas/600um/x_scan_slope_val_' + str(n) + '.mat', dict(x_scan_slope_val = self.slope_x))
                    sp.io.savemat('xy_scan_aberr_meas/600um/y_scan_slope_val_' + str(n) + '.mat', dict(y_scan_slope_val = self.slope_y))

                    # Close HDF5 file
                    data_file.close()

            # Scan multiple points across a line in both x and y directions for a given number of loops over a small FOV (100um)
            if config['zern_test']['small_flag']:

                # Initialise array for storing retrieved zernike coefficients
                self.zern_x = np.zeros([config['AO']['recon_coeff_num'], config['zern_test']['scan_num_x_small'], config['zern_test']['loop_num']])
                self.zern_y = np.zeros([config['AO']['recon_coeff_num'], config['zern_test']['scan_num_y_small'], config['zern_test']['loop_num']])

                # Initialise array for storing retrieved slope values
                self.slope_x = np.zeros([self.SB_settings['act_ref_cent_num'] * 2, config['zern_test']['scan_num_x_small'], config['zern_test']['loop_num']])
                self.slope_y = np.zeros([self.SB_settings['act_ref_cent_num'] * 2, config['zern_test']['scan_num_y_small'], config['zern_test']['loop_num']])

                # Generate relevant amounts of x/y scan voltages (normalised) for scanning across sample
                x_array_small = np.linspace(-config['zern_test']['x_amp_small'], config['zern_test']['x_amp_small'], config['zern_test']['scan_num_x_small'])
                y_array_small = np.linspace(-config['zern_test']['y_amp_small'], config['zern_test']['y_amp_small'], config['zern_test']['scan_num_y_small'])

                for n in range(config['zern_test']['run_num']):

                    # Initialise AO information parameter
                    self.AO_info = {'zern_test': {}}

                    # Create new datasets in HDF5 file to store closed-loop AO data and open file
                    get_dset(self.SB_settings, 'zern_test', flag = 0)
                    data_file = h5py.File('data_info.h5', 'a')
                    data_set_1 = data_file['AO_img']['zern_test']
                    data_set_2 = data_file['AO_info']['zern_test']
                    
                    # Reset deformable mirror
                    self.mirror.Reset()

                    for l in range(config['zern_test']['loop_num']):

                        # Reset scanner
                        self.scanner.ResetDevicePosition()

                        print('Small FOV run {} x-scan loop {}'.format(n + 1, l + 1))

                        for m in range(config['zern_test']['scan_num_x_small']):
                        
                            if self.loop:
                                
                                try:
                                    
                                    # Send voltages to scanner
                                    self.scanner.GoToDevicePosition(x_array_small[m], 0, 255, 10)
                                
                                    # Acquire S-H spots using camera
                                    AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                    AO_image = np.mean(AO_image_stack, axis = 2)

                                    # Image thresholding to remove background
                                    AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                    AO_image[AO_image < 0] = 0
                                    self.image.emit(AO_image)

                                    # Append image to list
                                    dset_append(data_set_1, 'real_AO_img', AO_image)

                                    # Calculate centroids of S-H spots
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2) 
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                    # Draw actual S-H spot centroids on image layer
                                    AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                    self.image.emit(AO_image)

                                    # Take tip\tilt off
                                    slope_x -= np.mean(slope_x)
                                    slope_y -= np.mean(slope_y)

                                    # Concatenate slopes into one slope matrix
                                    slope = (np.concatenate((slope_x, slope_y), axis = 1)).T
                                    self.slope_x[:,m,l] = slope[:,0]

                                    # Get detected zernike coefficients from slope matrix
                                    self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                                    # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                                    zern_err = self.zern_coeff_detect.copy()
                                    rms_zern = np.sqrt((zern_err ** 2).sum())
                                    self.zern_x[:,m,l] = zern_err[:,0]

                                    print('Full zernike root mean square error {} is {} um'.format(m + 1, rms_zern))                              

                                except Exception as e:
                                    print(e)
                            else:

                                self.done.emit()

                    for l in range(config['zern_test']['loop_num']):

                        # Reset scanner
                        self.scanner.ResetDevicePosition()

                        print('Small FOV run {} y-scan loop {}'.format(n + 1, l + 1))

                        for m in range(config['zern_test']['scan_num_y_small']):
                        
                            if self.loop:
                                
                                try:

                                    # Send voltages to scanner
                                    self.scanner.GoToDevicePosition(0, y_array_small[m], 255, 10)
                                
                                    # Acquire S-H spots using camera
                                    AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                    AO_image = np.mean(AO_image_stack, axis = 2)

                                    # Image thresholding to remove background
                                    AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                    AO_image[AO_image < 0] = 0
                                    self.image.emit(AO_image)

                                    # Append image to list
                                    dset_append(data_set_1, 'real_AO_img', AO_image)

                                    # Calculate centroids of S-H spots
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2) 
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                    # Draw actual S-H spot centroids on image layer
                                    AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                    self.image.emit(AO_image)

                                    # Take tip\tilt off
                                    slope_x -= np.mean(slope_x)
                                    slope_y -= np.mean(slope_y)

                                    # Concatenate slopes into one slope matrix
                                    slope = (np.concatenate((slope_x, slope_y), axis = 1)).T
                                    self.slope_y[:,m,l] = slope[:,0]

                                    # Get detected zernike coefficients from slope matrix
                                    self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                                    # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                                    zern_err = self.zern_coeff_detect.copy()
                                    rms_zern = np.sqrt((zern_err ** 2).sum())
                                    self.zern_y[:,m,l] = zern_err[:,0]

                                    print('Full zernike root mean square error {} is {} um'.format(m + 1, rms_zern))                              

                                except Exception as e:
                                    print(e)
                            else:

                                self.done.emit()

                    sp.io.savemat('xy_scan_aberr_meas/100um/x_scan_zern_coeff_' + str(n) + '.mat', dict(x_scan_zern_coeff = self.zern_x))
                    sp.io.savemat('xy_scan_aberr_meas/100um/y_scan_zern_coeff_' + str(n) + '.mat', dict(y_scan_zern_coeff = self.zern_y))
                    sp.io.savemat('xy_scan_aberr_meas/100um/x_scan_slope_val_' + str(n) + '.mat', dict(x_scan_slope_val = self.slope_x))
                    sp.io.savemat('xy_scan_aberr_meas/100um/y_scan_slope_val_' + str(n) + '.mat', dict(y_scan_slope_val = self.slope_y))

                    # Close HDF5 file
                    data_file.close()

            self.message.emit('\nProcess complete.')

            prev2 = time.perf_counter()
            print('Time for Zernike coefficient retrieval from different points along line scan is:', (prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['zern_test']['x_scan_zern_coeff'] = self.zern_x
                self.AO_info['zern_test']['y_scan_zern_coeff'] = self.zern_y
                self.AO_info['zern_test']['x_scan_slope_val'] = self.slope_x
                self.AO_info['zern_test']['y_scan_slope_val'] = self.slope_y

                self.info.emit(self.AO_info)
                self.write.emit()

            else:

                self.done.emit()

            # Finished closed-loop AO process
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)

    @Slot(object)
    def stop(self):
        self.loop = False
        self.log = False


