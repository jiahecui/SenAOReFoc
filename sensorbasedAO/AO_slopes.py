from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import logging
import sys
import os
import argparse
import time
import h5py
import numpy as np

import log
from config import config
from HDF5_dset import dset_append, get_dset, get_mat_dset
from image_acquisition import acq_image
from centroid_acquisition import acq_centroid
from gaussian_inf import inf
from common import fft_spot_from_phase
from zernike_phase import zern_phase
from reflectance_process import reflect_process

logger = log.get_logger(__name__)

class AO_Slopes(QObject):
    """
    Runs closed-loop AO using calibrated slope control matrix
    """
    start = Signal()
    write = Signal()
    done = Signal(object)
    done2 = Signal(object)
    error = Signal(object)
    image = Signal(object)
    message = Signal(object)
    info = Signal(object)

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
        
        # Initialise Zernike coefficient array
        self.zern_coeff = np.zeros(config['AO']['control_coeff_num'])

        # Get remote focusing settings on demand and initialise relevant parameters and arrays for closed-loop correction process
        if self.AO_settings['focus_enable'] == 1:
            self.focus_settings = settings['focusing_info']
            self.correct_num = int(self.focus_settings['step_num'])
        else:
            self.correct_num = 1
        self.loop_rms_slopes = np.zeros([config['AO']['loop_max'] + 1, self.correct_num])
        self.loop_rms_zern, self.loop_rms_zern_part = (np.zeros([config['AO']['loop_max'] + 1, self.correct_num]) for i in range(2))
        self.strehl, self.strehl_2 = (np.zeros([config['AO']['loop_max'] + 1, self.correct_num]) for i in range(2))

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_0']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_1']

        # Determine number of scan points during AO correction
        if config['dummy'] or config['AO']['scan_flag'] == 0:
            self.scan_num_x = 1
        elif config['AO']['scan_flag'] == 1:
            self.scan_num_x = config['AO']['scan_num_x']
            self.scan_slope_x = np.zeros([self.SB_settings['act_ref_cent_num'] * 2, config['AO']['scan_num_x']])

        super().__init__()

    def strehl_calc(self, phase):
        """
        Calculates the strehl ratio of a given phase profile
        """
        # Get meshgrid of coordinates within phase image
        coord_xx, coord_yy = self.get_coord_mesh(self.SB_settings['sensor_width'])

        # Get boolean pupil mask
        pupil_mask = self.get_pupil_mask(coord_xx, coord_yy)

        # Get average phase and phase deviation across pupil aperture
        phase_ave = np.mean(phase[pupil_mask])
        phase_std = np.std(phase[pupil_mask])
        phase_delta = (phase - phase_ave) * pupil_mask

        # print('Max and min values in phase before subtracting average phase: {}, {}'.format(np.amax(phase), np.amin(phase)))
        # print('Max and min values in phase after subtracting average phase: {}, {}'.format(np.amax(phase_delta), np.amin(phase_delta)))
        print('Average phase value is:', phase_ave)
        print('Standard deviation of phase is:', phase_std)

        # Calculate Strehl ratio estimated using only the statistics of the phase deviation, according to Mahajan
        phase_delta_2 = phase_delta ** 2 * pupil_mask
        sigma_2 = np.mean(phase_delta_2[pupil_mask])
        strehl = np.exp(-(2 * np.pi / config['AO']['lambda']) ** 2 * sigma_2)

        return strehl

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

        print('Max and min values in delta_phase are: {} um, {} um'.format(np.amax(delta_phase), np.amin(delta_phase)))

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
    def run1(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Normal closed-loop AO process using a FIXED GAIN, iterated until residual phase error is below value given by Marechel 
            criterion or iteration has reached maximum
            """
            # Initialise AO information parameter
            self.AO_info = {'slope_AO_1': {}}
        
            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'slope_AO_1', flag = 2)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['slope_AO_1']
            data_set_2 = data_file['AO_info']['slope_AO_1']

            self.message.emit('\nProcess started for closed-loop AO via slopes...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            prev1 = time.perf_counter()

            # Run closed-loop control until tolerance value or maximum loop iteration is reached
            for i in range(config['AO']['loop_max'] + 1):
                
                if self.loop:

                    try:

                        # Update mirror control voltages
                        if i == 0:

                            # Determine whether to generate Zernike modes using DM
                            if not config['dummy'] and config['AO']['zern_gen']:

                                # Retrieve input zernike coefficient array
                                zern_array_temp = np.array(self.SB_settings['zernike_array_test'])
                                zern_array = np.zeros(config['AO']['control_coeff_num'])
                                zern_array[:len(zern_array_temp)] = zern_array_temp

                                # Retrieve actuator voltages from zernike coefficient array
                                voltages = np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                    [:,:config['AO']['control_coeff_num']], zern_array))
                            else:

                                voltages[:] = config['DM']['vol_bias']
                        else:

                            voltages -= 0.5 * config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_slopes'], slope_err))

                            print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

                        if config['dummy']:

                            # Update phase profile and retrieve S-H spot image 
                            if i == 0:

                                # Option 1: Load real phase profile from .mat file
                                if config['real_phase']:

                                    # Retrieve real phase profile
                                    phase_init = get_mat_dset(self.SB_settings, flag = 1)

                                # Option 2: Generate real zernike phase profile using DM control matrix
                                elif config['real_zernike']:

                                    # Retrieve input zernike coefficient array
                                    zern_array_temp = np.array(self.SB_settings['zernike_array_test'])

                                    # Pad zernike coefficient array to length of control_coeff_num
                                    zern_array = np.zeros(config['AO']['control_coeff_num'])
                                    zern_array[:len(zern_array_temp)] = zern_array_temp

                                    # Retrieve actuator voltages from zernike coefficient array
                                    voltages = np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                        [:,:config['AO']['control_coeff_num']], zern_array))
                                    
                                    # Generate zernike phase profile from DM
                                    phase_init = self.phase_calc(voltages)

                                    # Check whether need to incorporate sample reflectance process
                                    if config['reflect_on'] == 1:
                                        phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam)
                                    
                                # Option 3: Generate ideal zernike phase profile
                                else:
                                    
                                    # Retrieve input zernike coefficient array
                                    zern_array =  self.SB_settings['zernike_array_test']
                                    
                                    # Generate ideal zernike phase profile
                                    phase_init = zern_phase(self.SB_settings, zern_array)

                                    # Check whether need to incorporate sample reflectance process
                                    if config['reflect_on'] == 1:
                                        phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam)

                                # Display initial phase
                                self.image.emit(phase_init)

                                print('\nMax and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase_init), np.amin(phase_init)))

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

                                print('Max and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase), np.amin(phase)))

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
                        
                            # Acquire S-H spots using camera and append to list
                            AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                            AO_image = np.mean(AO_image_stack, axis = 2)
                            dset_append(data_set_1, 'real_AO_img', AO_image)

                        # Image thresholding to remove background
                        AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                        AO_image[AO_image < 0] = 0
                        self.image.emit(AO_image)

                        # Calculate centroids of S-H spots
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 4)
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                        # print('slope_x:', slope_x)
                        # print('slope_y:', slope_y)

                        # Draw actual S-H spot centroids on image layer
                        AO_image.ravel()[act_cent_coord.astype(int)] = 0
                        self.image.emit(AO_image)

                        # Concatenate slopes into one slope matrix
                        slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                        # Get phase residual (slope residual error) and calculate root mean square (rms) error
                        slope_err = slope.copy()
                        rms_slope = np.sqrt((slope_err ** 2).mean())
                        self.loop_rms_slopes[i] = rms_slope

                        print('Slope root mean square error {} is {} pixels'.format(i, rms_slope))

                        # Get detected zernike coefficients from slope matrix
                        self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                        # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                        zern_err = self.zern_coeff_detect.copy()
                        zern_err_part = self.zern_coeff_detect.copy()
                        zern_err_part[[0, 1], 0] = 0
                        rms_zern = np.sqrt((zern_err ** 2).sum())
                        rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                        self.loop_rms_zern[i] = rms_zern
                        self.loop_rms_zern_part[i] = rms_zern_part

                        strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)
                        self.strehl[i] = strehl
                        if config['dummy']:
                            strehl_2 = self.strehl_calc(phase)
                            self.strehl_2[i] = strehl_2

                        print('Full zernike root mean square error {} is {} um'.format(i, rms_zern))
                        print('Partial zernike root mean square error {} is {} um'.format(i, rms_zern_part))                        
                        print('Strehl ratio {} from rms_zern_part is: {}'.format(i, strehl))
                        if config['dummy']:
                            print('Strehl ratio {} from phase profile is: {} \n'.format(i, strehl_2))                       

                        # Append data to list
                        if config['dummy']:
                            dset_append(data_set_2, 'dummy_spot_slope_x', slope_x)
                            dset_append(data_set_2, 'dummy_spot_slope_y', slope_y)
                            dset_append(data_set_2, 'dummy_spot_slope', slope)
                            dset_append(data_set_2, 'dummy_spot_slope_err', slope_err)
                            dset_append(data_set_2, 'dummy_spot_zern_err', zern_err)
                        else:
                            dset_append(data_set_2, 'real_spot_slope_x', slope_x)
                            dset_append(data_set_2, 'real_spot_slope_y', slope_y)
                            dset_append(data_set_2, 'real_spot_slope', slope)
                            dset_append(data_set_2, 'real_spot_slope_err', slope_err)
                            dset_append(data_set_2, 'real_spot_zern_err', zern_err)

                        # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                        if strehl >= config['AO']['tolerance_fact_strehl']:
                            break                 

                    except Exception as e:
                        print(e)
                else:

                    self.done.emit(1)

            # Close HDF5 file
            data_file.close()

            self.message.emit('\nProcess complete.')
            print('Final root mean square error of detected wavefront is: {} um'.format(rms_zern))

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is: {} s'.format(prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['slope_AO_1']['loop_num'] = i
                self.AO_info['slope_AO_1']['residual_phase_err_slopes'] = self.loop_rms_slopes
                self.AO_info['slope_AO_1']['residual_phase_err_zern'] = self.loop_rms_zern
                self.AO_info['slope_AO_1']['residual_phase_err_zern_part'] = self.loop_rms_zern_part
                self.AO_info['slope_AO_1']['strehl_ratio'] = self.strehl
                if config['dummy']:
                    self.AO_info['slope_AO_1']['strehl_ratio_2'] = self.strehl_2

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit(1)

            # Finished closed-loop AO process
            self.done.emit(1)

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
            Closed-loop AO process to deal with obscured S-H spots using a FIXED GAIN, iterated until residual phase error is below value 
            given by Marechel criterion or iteration has reached maximum
            """
            # Initialise AO information parameter
            self.AO_info = {'slope_AO_2': {}}

            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'slope_AO_2', flag = 2)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['slope_AO_2']
            data_set_2 = data_file['AO_info']['slope_AO_2']

            self.message.emit('\nProcess started for closed-loop AO via slopes with obscured subapertures...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            prev1 = time.perf_counter()

            # Run closed-loop control until tolerance value or maximum loop iteration is reached
            for i in range(config['AO']['loop_max'] + 1):
                
                if self.loop:

                    try:

                        # Update mirror control voltages
                        if i == 0:

                            # Determine whether to generate Zernike modes using DM
                            if not config['dummy'] and config['AO']['zern_gen']:

                                # Retrieve input zernike coefficient array
                                zern_array_temp = np.array(self.SB_settings['zernike_array_test'])
                                zern_array = np.zeros(config['AO']['control_coeff_num'])
                                zern_array[:len(zern_array_temp)] = zern_array_temp

                                # Retrieve actuator voltages from zernike coefficient array
                                voltages = np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                    [:,:config['AO']['control_coeff_num']], zern_array))
                            else:

                                voltages[:] = config['DM']['vol_bias']
                        else:

                            voltages -= 0.5 * config['AO']['loop_gain'] * np.ravel(np.dot(control_matrix_slopes, slope_err))

                            print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

                        if config['dummy']:
                            
                            # Update phase profile and retrieve S-H spot image 
                            if i == 0:

                                # Option 1: Load real phase profile from .mat file
                                if config['real_phase']:

                                    # Retrieve real phase profile
                                    phase_init = get_mat_dset(self.SB_settings, flag = 1)

                                # Option 2: Generate real zernike phase profile using DM control matrix
                                elif config['real_zernike']:

                                    # Retrieve input zernike coefficient array
                                    zern_array_temp = np.array(self.SB_settings['zernike_array_test'])

                                    # Pad zernike coefficient array to length of control_coeff_num
                                    zern_array = np.zeros(config['AO']['control_coeff_num'])
                                    zern_array[:len(zern_array_temp)] = zern_array_temp

                                    # Retrieve actuator voltages from zernike coefficient array
                                    voltages = np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                        [:,:config['AO']['control_coeff_num']], zern_array))
                                    
                                    # Generate zernike phase profile from DM
                                    phase_init = self.phase_calc(voltages)

                                    # Check whether need to incorporate sample reflectance process
                                    if config['reflect_on'] == 1:
                                        phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam)
                                    
                                # Option 3: Generate ideal zernike phase profile
                                else:
                                    
                                    # Retrieve input zernike coefficient array
                                    zern_array =  self.SB_settings['zernike_array_test']
                                    
                                    # Generate ideal zernike phase profile
                                    phase_init = zern_phase(self.SB_settings, zern_array) 

                                    # Check whether need to incorporate sample reflectance process
                                    if config['reflect_on'] == 1:
                                        phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam)

                                # Display initial phase
                                self.image.emit(phase_init)

                                print('\nMax and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase_init), np.amin(phase_init)))

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

                                print('Max and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase), np.amin(phase)))

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
                        
                            # Acquire S-H spots using camera and append to list
                            AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                            AO_image = np.mean(AO_image_stack, axis = 2)
                            dset_append(data_set_1, 'real_AO_img', AO_image)

                        # Image thresholding to remove background
                        AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                        AO_image[AO_image < 0] = 0
                        self.image.emit(AO_image)

                        # Detect obscured subapertures while calculating centroids of S-H spots
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 6)
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                        # print('slope_x:', slope_x)
                        # print('slope_y:', slope_y)

                        # Remove corresponding elements from slopes and rows from influence function matrix
                        if config['dummy'] == 1:
                            index_remove = np.where(slope_x + self.SB_settings['act_ref_cent_coord_x'].astype(int) + 1 == 0)[1]
                        else:
                            index_remove = np.where(slope_x + self.SB_settings['act_ref_cent_coord_x'] == 0)[1]
                        print('Number of obscured subapertures:', np.size(index_remove))
                        print('Index of obscured subapertures:', index_remove)
                        index_remove_inf = np.concatenate((index_remove, index_remove + self.SB_settings['act_ref_cent_num']), axis = None)
                        # print('Shape index_remove_inf:', np.shape(index_remove_inf))
                        # print('index_remove_inf:', index_remove_inf)
                        slope_x = np.delete(slope_x, index_remove, axis = 1)
                        slope_y = np.delete(slope_y, index_remove, axis = 1)
                        # print('Shape slope_x:', np.shape(slope_x))
                        # print('Shape slope_y:', np.shape(slope_y))
                        act_cent_coord = np.delete(act_cent_coord, index_remove, axis = None)
                        # print('Shape act_cent_coord:', np.shape(act_cent_coord))
                        inf_matrix_slopes = np.delete(self.mirror_settings['inf_matrix_slopes'].copy(), index_remove_inf, axis = 0)
                        # print('Shape inf_matrix_slopes:', np.shape(inf_matrix_slopes))
                        conv_matrix = np.delete(self.mirror_settings['conv_matrix'].copy(), index_remove_inf, axis = 1)
                        # print('Shape conv_matrix:', np.shape(conv_matrix))

                        # Draw actual S-H spot centroids on image layer
                        AO_image.ravel()[act_cent_coord.astype(int)] = 0
                        self.image.emit(AO_image)

                        # Calculate singular value decomposition of modified influence function matrix
                        u, s, vh = np.linalg.svd(inf_matrix_slopes, full_matrices = False)
                        # print('u: {}, s: {}, vh: {}'.format(u, s, vh))
                        # print('The shapes of u, s, and vh are: {}, {}, and {}'.format(np.shape(u), np.shape(s), np.shape(vh)))

                        # Recalculate pseudo-inverse of modified influence function matrix to get new control matrix
                        control_matrix_slopes = np.linalg.pinv(inf_matrix_slopes)
                        # print('Shape of new control matrix is:', np.shape(control_matrix_slopes))

                        # Concatenate slopes into one slope matrix
                        slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                        # Get phase residual (slope residual error) and calculate root mean square (rms) error
                        slope_err = slope.copy()
                        rms_slope = np.sqrt((slope_err ** 2).mean())
                        self.loop_rms_slopes[i] = rms_slope

                        print('Slope root mean square error {} is {} pixels'.format(i, rms_slope))

                        # Get detected zernike coefficients from slope matrix
                        self.zern_coeff_detect = np.dot(conv_matrix, slope)

                        # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                        zern_err = self.zern_coeff_detect.copy()
                        zern_err_part = self.zern_coeff_detect.copy()
                        zern_err_part[[0, 1], 0] = 0
                        rms_zern = np.sqrt((zern_err ** 2).sum())
                        rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                        self.loop_rms_zern[i] = rms_zern
                        self.loop_rms_zern_part[i] = rms_zern_part

                        strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)
                        self.strehl[i] = strehl
                        if config['dummy']:
                            strehl_2 = self.strehl_calc(phase)
                            self.strehl_2[i] = strehl_2

                        print('Full zernike root mean square error {} is {} um'.format(i, rms_zern))
                        print('Partial zernike root mean square error {} is {} um'.format(i, rms_zern_part))                        
                        print('Strehl ratio {} from rms_zern_part is: {}'.format(i, strehl))
                        if config['dummy']:
                            print('Strehl ratio {} from phase profile is: {} \n'.format(i, strehl_2))                            

                        # Append data to list
                        if config['dummy']:
                            dset_append(data_set_2, 'dummy_spot_zern_err', zern_err)
                        else:
                            dset_append(data_set_2, 'real_spot_zern_err', zern_err)

                        # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                        if strehl >= config['AO']['tolerance_fact_strehl']:
                            break                 

                    except Exception as e:
                        print(e)
                else:

                    self.done.emit(2)

            # Close HDF5 file
            data_file.close()

            self.message.emit('\nProcess complete.')
            print('Final root mean square error of detected wavefront is: {} um'.format(rms_zern))

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is: {} s'.format(prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['slope_AO_2']['loop_num'] = i
                self.AO_info['slope_AO_2']['residual_phase_err_slopes'] = self.loop_rms_slopes
                self.AO_info['slope_AO_2']['residual_phase_err_zern'] = self.loop_rms_zern
                self.AO_info['slope_AO_2']['residual_phase_err_zern_part'] = self.loop_rms_zern_part
                self.AO_info['slope_AO_2']['strehl_ratio'] = self.strehl
                if config['dummy']:
                    self.AO_info['slope_AO_2']['strehl_ratio_2'] = self.strehl_2

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit(2)

            # Finished closed-loop AO process
            self.done.emit(2)

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
            Closed-loop AO process to handle partial correction using a FIXED GAIN, iterated until residual phase error is below value given by Marechel 
            criterion or iteration has reached maximum
            """
            # Initialise AO information parameter
            self.AO_info = {'slope_AO_3': {}}

            # Calculate modified influence function with partial correction (suppressing tip, tilt, and defocus)
            inf_matrix_slopes = np.concatenate((self.mirror_settings['inf_matrix_slopes'], config['AO']['suppress_gain'] * \
                self.mirror_settings['inf_matrix_zern'][[0, 1, 3], :]), axis = 0)

            # Calculate singular value decomposition of modified influence function matrix
            u, s, vh = np.linalg.svd(inf_matrix_slopes, full_matrices = False)
            # print('u: {}, s: {}, vh: {}'.format(u, s, vh))
            # print('The shapes of u, s, and vh are: {}, {}, and {}'.format(np.shape(u), np.shape(s), np.shape(vh)))

            # Calculate pseudo-inverse of modified influence function matrix to get new control matrix
            control_matrix_slopes = np.linalg.pinv(inf_matrix_slopes)
            # print('Shape of new control matrix is:', np.shape(control_matrix_slopes))
        
            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'slope_AO_3', flag = 2)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['slope_AO_3']
            data_set_2 = data_file['AO_info']['slope_AO_3']

            if self.AO_settings['focus_enable'] == 0:
                self.message.emit('\nProcess started for closed-loop AO via slopes with partial correction...')
            else:
                self.message.emit('\nProcess started for remote focusing + closed-loop AO via slopes with partial correction...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            prev1 = time.perf_counter()

            # Run correction for each focus depth
            for j in range(self.correct_num):

                # Retrieve voltages for remote focusing component
                if self.AO_settings['focus_enable'] == 1:
                    if self.focus_settings['focus_mode_flag'] == 0:
                        RF_index = self.focus_settings['focus_depth_defoc'] // config['RF_calib']['step_incre']
                        voltages_defoc = np.ravel(self.mirror_settings['remote_focus_voltages'][:, RF_index])
                    else:
                        RF_index = self.focus_settings['start_depth_defoc'] // config['RF_calib']['step_incre'] \
                            + self.focus_settings['step_incre_defoc'] // config['RF_calib']['step_incre'] * j
                        voltages_defoc = np.ravel(self.mirror_settings['remote_focus_voltages'][:, RF_index])
                else:
                    voltages_defoc = 0

                # Run closed-loop control until tolerance value or maximum loop iteration is reached
                for i in range(config['AO']['loop_max'] + 1):
                    
                    if self.loop:

                        try:

                            # Update mirror control voltages
                            if i == 0:

                                # Determine whether to generate Zernike modes using DM
                                if not config['dummy'] and config['AO']['zern_gen']:

                                    # Retrieve input zernike coefficient array
                                    zern_array_temp = np.array(self.SB_settings['zernike_array_test'])
                                    zern_array = np.zeros(config['AO']['control_coeff_num'])
                                    zern_array[:len(zern_array_temp)] = zern_array_temp

                                    # Retrieve actuator voltages from zernike coefficient array
                                    voltages = np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                        [:,:config['AO']['control_coeff_num']], zern_array)) + voltages_defoc
                                else:

                                    voltages[:] = config['DM']['vol_bias'] + voltages_defoc
                            else:

                                voltages -= 0.5 * config['AO']['loop_gain'] * np.ravel(np.dot(control_matrix_slopes, slope_err))

                                print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

                            if config['dummy']:
                                
                                # Update phase profile and retrieve S-H spot image 
                                if i == 0:

                                    # Option 1: Load real phase profile from .mat file
                                    if config['real_phase']:

                                        # Retrieve real phase profile
                                        phase_init = get_mat_dset(self.SB_settings, flag = 1)

                                        # Generate defocus phase profile
                                        phase_defoc = self.phase_calc(voltages_defoc)

                                        # Apply defocus to real phase profile
                                        phase_init += phase_defoc

                                    # Option 2: Generate real zernike phase profile using DM control matrix
                                    elif config['real_zernike']:

                                        # Retrieve input zernike coefficient array
                                        zern_array_temp = self.SB_settings['zernike_array_test']

                                        # Pad zernike coefficient array to length of control_coeff_num
                                        zern_array = np.zeros(config['AO']['control_coeff_num'])
                                        zern_array[:len(zern_array_temp)] = zern_array_temp

                                        # Retrieve actuator voltages from zernike coefficient array + defocus component 
                                        voltages = np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                            [:,:config['AO']['control_coeff_num']], zern_array)) + voltages_defoc
                                        
                                        # Generate zernike + defocus phase profile from DM
                                        phase_init = self.phase_calc(voltages)

                                        # Check whether need to incorporate sample reflectance process
                                        if config['reflect_on'] == 1:
                                            phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam)
                                        
                                    # Option 3: Generate ideal zernike phase profile
                                    else:
                                        
                                        # Retrieve input zernike coefficient array + defocus component
                                        zern_array_temp =  self.SB_settings['zernike_array_test']
                                        if len(zern_array_temp) >= 4:
                                            zern_array_temp[3] += self.zern_coeff[3]
                                            zern_array = zern_array_temp
                                        else:
                                            zern_array = np.zeros(4)
                                            zern_array[:len(zern_array_temp)] = zern_array_temp
                                            zern_array[3] = self.zern_coeff[3]
                                        
                                        # Generate ideal zernike phase profile
                                        phase_init = zern_phase(self.SB_settings, zern_array) 

                                        # Check whether need to incorporate sample reflectance process
                                        if config['reflect_on'] == 1:
                                            phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam)

                                    # Display initial phase
                                    self.image.emit(phase_init)

                                    print('\nMax and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase_init), np.amin(phase_init)))

                                    # Get simulated S-H spots and append to list
                                    AO_image, spot_cent_x, spot_cent_y = fft_spot_from_phase(self.SB_settings, phase_init)
                                    dset_append(data_set_1, 'dummy_AO_img', AO_image)
                                    dset_append(data_set_1, 'dummy_spot_cent_x', spot_cent_x)
                                    dset_append(data_set_1, 'dummy_spot_cent_y', spot_cent_y)

                                    phase = phase_init.copy()

                                else:

                                    # Calculate phase profile introduced by DM
                                    delta_phase = self.phase_calc(voltages - voltages_defoc)

                                    # Update phase data
                                    phase = phase_init - delta_phase

                                    # Display corrected phase
                                    self.image.emit(phase)

                                    print('Max and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase), np.amin(phase)))

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
                            
                                # Acquire S-H spots using camera and append to list
                                AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                AO_image = np.mean(AO_image_stack, axis = 2)
                                dset_append(data_set_1, 'real_AO_img', AO_image)

                            # Image thresholding to remove background
                            AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                            AO_image[AO_image < 0] = 0
                            self.image.emit(AO_image)

                            # Calculate centroids of S-H spots
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 8)
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                            # print('slope_x:', slope_x)
                            # print('slope_y:', slope_y)

                            # Draw actual S-H spot centroids on image layer
                            AO_image.ravel()[act_cent_coord.astype(int)] = 0
                            self.image.emit(AO_image)

                            # Concatenate slopes into one slope matrix
                            slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                            # Get phase residual (slope residual error) and calculate root mean square (rms) error
                            slope_err = slope.copy()
                            rms_slope = np.sqrt((slope_err ** 2).mean())
                            self.loop_rms_slopes[i,j] = rms_slope

                            print('Slope root mean square error {} is {} pixels'.format(i, rms_slope))

                            # Get detected zernike coefficients from slope matrix
                            self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                            # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                            zern_err = self.zern_coeff_detect.copy()
                            zern_err_part = self.zern_coeff_detect.copy()
                            zern_err_part[[0, 1, 3], 0] = 0
                            rms_zern = np.sqrt((zern_err ** 2).sum())
                            rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                            self.loop_rms_zern[i,j] = rms_zern
                            self.loop_rms_zern_part[i,j] = rms_zern_part

                            strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)
                            self.strehl[i,j] = strehl
                            if config['dummy']:
                                strehl_2 = self.strehl_calc(phase)
                                self.strehl_2[i,j] = strehl_2

                            print('Full zernike root mean square error {} is {} um'.format(i, rms_zern))
                            print('Partial zernike root mean square error {} is {} um'.format(i, rms_zern_part))                        
                            print('Strehl ratio {} from rms_zern_part is: {}'.format(i, strehl))
                            if config['dummy']:
                                print('Strehl ratio {} from phase profile is: {} \n'.format(i, strehl_2))                              

                            # Append data to list
                            if config['dummy']:
                                dset_append(data_set_2, 'dummy_spot_slope_x', slope_x)
                                dset_append(data_set_2, 'dummy_spot_slope_y', slope_y)
                                dset_append(data_set_2, 'dummy_spot_slope', slope)
                                dset_append(data_set_2, 'dummy_spot_slope_err', slope_err)
                                dset_append(data_set_2, 'dummy_spot_zern_err', zern_err)
                            else:
                                dset_append(data_set_2, 'real_spot_slope_x', slope_x)
                                dset_append(data_set_2, 'real_spot_slope_y', slope_y)
                                dset_append(data_set_2, 'real_spot_slope', slope)
                                dset_append(data_set_2, 'real_spot_slope_err', slope_err)
                                dset_append(data_set_2, 'real_spot_zern_err', zern_err)

                            # Append zeros to end of slope error array to ensure dimension is consistent with new control matrix
                            slope_err = np.append(slope_err, np.zeros([3, 1]), axis = 0)

                            # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                            if strehl >= config['AO']['tolerance_fact_strehl']:
                                break                 

                        except Exception as e:
                            print(e)
                    else:

                        if self.AO_settings['focus_enable'] == 0:
                            self.done.emit(3)
                        elif self.focus_settings['focus_mode_flag'] == 0:
                            self.done2.emit(0)
                        elif self.focus_settings['focus_mode_flag'] == 1:
                            self.done2.emit(1)

                print('Final root mean square error of detected wavefront is: {} um'.format(rms_zern))

            # Close HDF5 file
            data_file.close()

            self.message.emit('\nProcess complete.')

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is: {} s'.format(prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['slope_AO_3']['loop_num'] = i
                self.AO_info['slope_AO_3']['residual_phase_err_slopes'] = self.loop_rms_slopes
                self.AO_info['slope_AO_3']['residual_phase_err_zern'] = self.loop_rms_zern
                self.AO_info['slope_AO_3']['residual_phase_err_zern_part'] = self.loop_rms_zern_part
                self.AO_info['slope_AO_3']['strehl_ratio'] = self.strehl
                if config['dummy']:
                    self.AO_info['slope_AO_3']['strehl_ratio_2'] = self.strehl_2

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                if self.AO_settings['focus_enable'] == 0:
                    self.done.emit(3)
                elif self.focus_settings['focus_mode_flag'] == 0:
                    self.done2.emit(0)
                elif self.focus_settings['focus_mode_flag'] == 1:
                    self.done2.emit(1)

            # Finished closed-loop AO process
            if self.AO_settings['focus_enable'] == 0:
                self.done.emit(3)
            elif self.focus_settings['focus_mode_flag'] == 0:
                self.done2.emit(0)
            elif self.focus_settings['focus_mode_flag'] == 1:
                self.done2.emit(1)

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
            Closed-loop AO process to handle both obscured S-H spots and partial correction using a FIXED GAIN, iterated until residual 
            phase error is below value given by Marechel criterion or iteration has reached maximum
            """
            # Initialise AO information parameter
            self.AO_info = {'slope_AO_full': {}}

            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'slope_AO_full', flag = 2)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['slope_AO_full']
            data_set_2 = data_file['AO_info']['slope_AO_full']

            if self.AO_settings['focus_enable'] == 0:
                self.message.emit('\nProcess started for full closed-loop AO via slopes...')
            else:
                self.message.emit('\nProcess started for remote focusing + full closed-loop AO via slopes...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            prev1 = time.perf_counter()

            # Run correction for each focus depth
            for j in range(self.correct_num):

                # Retrieve voltages for remote focusing component
                if self.AO_settings['focus_enable'] == 1:
                    if self.focus_settings['focus_mode_flag'] == 0:
                        RF_index = self.focus_settings['focus_depth_defoc'] // config['RF_calib']['step_incre']
                        voltages_defoc = np.ravel(self.mirror_settings['remote_focus_voltages'][:, RF_index])
                    else:
                        RF_index = self.focus_settings['start_depth_defoc'] // config['RF_calib']['step_incre'] \
                            + self.focus_settings['step_incre_defoc'] // config['RF_calib']['step_incre'] * j
                        voltages_defoc = np.ravel(self.mirror_settings['remote_focus_voltages'][:, RF_index])
                else:
                    voltages_defoc = 0

                # Run closed-loop control until tolerance value or maximum loop iteration is reached
                for i in range(config['AO']['loop_max'] + 1):
                    
                    if self.loop:

                        try:

                            # Update mirror control voltages
                            if i == 0:

                                # Determine whether to generate Zernike modes using DM
                                if not config['dummy'] and config['AO']['zern_gen']:

                                    # Retrieve input zernike coefficient array
                                    zern_array_temp = np.array(self.SB_settings['zernike_array_test'])
                                    zern_array = np.zeros(config['AO']['control_coeff_num'])
                                    zern_array[:len(zern_array_temp)] = zern_array_temp

                                    # Retrieve actuator voltages from zernike coefficient array
                                    voltages = np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                        [:,:config['AO']['control_coeff_num']], zern_array)) + voltages_defoc
                                else:

                                    voltages[:] = config['DM']['vol_bias'] + voltages_defoc
                            else:
                                
                                voltages -= 0.5 * config['AO']['loop_gain'] * np.ravel(np.dot(control_matrix_slopes, slope_err))

                                print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

                            if config['dummy']:
                                
                                # Update phase profile and retrieve S-H spot image 
                                if i == 0:

                                    # Option 1: Load real phase profile from .mat file
                                    if config['real_phase']:

                                        # Retrieve real phase profile
                                        phase_init = get_mat_dset(self.SB_settings, flag = 1)

                                        # Generate defocus phase profile
                                        phase_defoc = self.phase_calc(voltages_defoc)

                                        # Apply defocus to real phase profile
                                        phase_init += phase_defoc

                                    # Option 2: Generate real zernike phase profile using DM control matrix
                                    elif config['real_zernike']:

                                        # Retrieve input zernike coefficient array
                                        zern_array_temp = self.SB_settings['zernike_array_test']

                                        # Pad zernike coefficient array to length of control_coeff_num
                                        zern_array = np.zeros(config['AO']['control_coeff_num'])
                                        zern_array[:len(zern_array_temp)] = zern_array_temp

                                        # Retrieve actuator voltages from zernike coefficient array + defocus component 
                                        voltages = np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                            [:,:config['AO']['control_coeff_num']], zern_array)) + voltages_defoc
                                        
                                        # Generate zernike + defocus phase profile from DM
                                        phase_init = self.phase_calc(voltages)

                                        # Check whether need to incorporate sample reflectance process
                                        if config['reflect_on'] == 1:
                                            phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam)
                                        
                                    # Option 3: Generate ideal zernike phase profile
                                    else:
                                        
                                        # Retrieve input zernike coefficient array + defocus component
                                        zern_array_temp =  self.SB_settings['zernike_array_test']
                                        if len(zern_array_temp) >= 4:
                                            zern_array_temp[3] += self.zern_coeff[3]
                                            zern_array = zern_array_temp
                                        else:
                                            zern_array = np.zeros(4)
                                            zern_array[:len(zern_array_temp)] = zern_array_temp
                                            zern_array[3] = self.zern_coeff[3]
                                        
                                        # Generate ideal zernike phase profile
                                        phase_init = zern_phase(self.SB_settings, zern_array) 

                                        # Check whether need to incorporate sample reflectance process
                                        if config['reflect_on'] == 1:
                                            phase_init = reflect_process(self.SB_settings, phase_init, self.pupil_diam)

                                    # Display initial phase
                                    self.image.emit(phase_init)

                                    print('\nMax and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase_init), np.amin(phase_init)))

                                    # Get simulated S-H spots and append to list
                                    AO_image, spot_cent_x, spot_cent_y = fft_spot_from_phase(self.SB_settings, phase_init)
                                    dset_append(data_set_1, 'dummy_AO_img', AO_image)
                                    dset_append(data_set_1, 'dummy_spot_cent_x', spot_cent_x)
                                    dset_append(data_set_1, 'dummy_spot_cent_y', spot_cent_y)

                                    phase = phase_init.copy()

                                else:

                                    # Calculate phase profile introduced by DM
                                    delta_phase = self.phase_calc(voltages - voltages_defoc)

                                    # Update phase data
                                    phase = phase_init - delta_phase

                                    # Display corrected phase
                                    self.image.emit(phase)

                                    print('Max and min values of phase {} are: {} um, {} um'.format(i, np.amax(phase), np.amin(phase)))

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
                            
                                # Acquire S-H spots using camera and append to list
                                AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                AO_image = np.mean(AO_image_stack, axis = 2)
                                dset_append(data_set_1, 'real_AO_img', AO_image)

                            # Image thresholding to remove background
                            AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                            AO_image[AO_image < 0] = 0
                            self.image.emit(AO_image)

                            # Calculate centroids of S-H spots
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 10)
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                            # print('slope_x:', slope_x)
                            # print('slope_y:', slope_y)

                            # Remove corresponding elements from slopes and rows from influence function matrix
                            if config['dummy'] == 1:
                                index_remove = np.where(slope_x + self.SB_settings['act_ref_cent_coord_x'].astype(int) + 1 == 0)[1]
                            else:
                                index_remove = np.where(slope_x + self.SB_settings['act_ref_cent_coord_x'] == 0)[1]
                            print('Number of obscured subapertures:', np.shape(index_remove))
                            print('Index of obscured subapertures:', index_remove)
                            index_remove_inf = np.concatenate((index_remove, index_remove + self.SB_settings['act_ref_cent_num']), axis = None)
                            # print('Shape index_remove_inf:', np.shape(index_remove_inf))
                            # print('index_remove_inf:', index_remove_inf)
                            slope_x = np.delete(slope_x, index_remove, axis = 1)
                            slope_y = np.delete(slope_y, index_remove, axis = 1)
                            # print('Shape slope_x:', np.shape(slope_x))
                            # print('Shape slope_y:', np.shape(slope_y))
                            act_cent_coord = np.delete(act_cent_coord, index_remove, axis = None)
                            # print('Shape act_cent_coord:', np.shape(act_cent_coord))
                            inf_matrix_slopes = np.delete(self.mirror_settings['inf_matrix_slopes'].copy(), index_remove_inf, axis = 0)
                            # print('Shape inf_matrix_slopes:', np.shape(inf_matrix_slopes))
                            conv_matrix = np.delete(self.mirror_settings['conv_matrix'].copy(), index_remove_inf, axis = 1)
                            # print('Shape conv_matrix:', np.shape(conv_matrix))

                            # Draw actual S-H spot centroids on image layer
                            AO_image.ravel()[act_cent_coord.astype(int)] = 0
                            self.image.emit(AO_image)

                            # Calculate modified influence function with partial correction (suppressing tip, tilt, and defocus)
                            inf_matrix_slopes = np.concatenate((inf_matrix_slopes, config['AO']['suppress_gain'] * \
                                self.mirror_settings['inf_matrix_zern'][[0, 1, 3], :]), axis = 0)

                            # Calculate singular value decomposition of modified influence function matrix
                            u, s, vh = np.linalg.svd(inf_matrix_slopes, full_matrices = False)
                            # print('u: {}, s: {}, vh: {}'.format(u, s, vh))
                            # print('The shapes of u, s, and vh are: {}, {}, and {}'.format(np.shape(u), np.shape(s), np.shape(vh)))

                            # Recalculate pseudo-inverse of modified influence function matrix to get new control matrix
                            control_matrix_slopes = np.linalg.pinv(inf_matrix_slopes)
                            # print('Shape of new control matrix is:', np.shape(control_matrix_slopes))

                            # Concatenate slopes into one slope matrix
                            slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                            # Get phase residual (slope residual error) and calculate root mean square (rms) error
                            slope_err = slope.copy()
                            rms_slope = np.sqrt((slope_err ** 2).mean())
                            self.loop_rms_slopes[i,j] = rms_slope

                            print('Slope root mean square error {} is {} pixels'.format(i, rms_slope))

                            # Get detected zernike coefficients from slope matrix
                            self.zern_coeff_detect = np.dot(conv_matrix, slope)

                            # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                            zern_err = self.zern_coeff_detect.copy()
                            zern_err_part = self.zern_coeff_detect.copy()
                            zern_err_part[[0, 1, 3], 0] = 0
                            rms_zern = np.sqrt((zern_err ** 2).sum())
                            rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                            self.loop_rms_zern[i,j] = rms_zern
                            self.loop_rms_zern_part[i,j] = rms_zern_part

                            strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)
                            self.strehl[i,j] = strehl
                            if config['dummy']:
                                strehl_2 = self.strehl_calc(phase)
                                self.strehl_2[i,j] = strehl_2

                            print('Full zernike root mean square error {} is {} um'.format(i, rms_zern))
                            print('Partial zernike root mean square error {} is {} um'.format(i, rms_zern_part))                        
                            print('Strehl ratio {} from rms_zern_part is: {}'.format(i, strehl))
                            if config['dummy']:
                                print('Strehl ratio {} from phase profile is: {} \n'.format(i, strehl_2))                        
                            
                            # Append data to list
                            if config['dummy']:
                                dset_append(data_set_2, 'dummy_spot_zern_err', zern_err)
                            else:
                                dset_append(data_set_2, 'real_spot_zern_err', zern_err)

                            # Append zeros to end of slope error array to ensure dimension is consistent with new control matrix
                            slope_err = np.append(slope_err, np.zeros([3, 1]), axis = 0)

                            # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                            if strehl >= config['AO']['tolerance_fact_strehl']:
                                break                 

                        except Exception as e:
                            print(e)
                    else:

                        if self.AO_settings['focus_enable'] == 0:
                            self.done.emit(4)
                        elif self.focus_settings['focus_mode_flag'] == 0:
                            self.done2.emit(0)
                        elif self.focus_settings['focus_mode_flag'] == 1:
                            self.done2.emit(1)

                print('Final root mean square error of detected wavefront is: {} um'.format(rms_zern))

            # Close HDF5 file
            data_file.close()

            self.message.emit('\nProcess complete.')

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is: {} s'.format(prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['slope_AO_full']['loop_num'] = i
                self.AO_info['slope_AO_full']['residual_phase_err_slopes'] = self.loop_rms_slopes
                self.AO_info['slope_AO_full']['residual_phase_err_zern'] = self.loop_rms_zern
                self.AO_info['slope_AO_full']['residual_phase_err_zern_part'] = self.loop_rms_zern_part
                self.AO_info['slope_AO_full']['strehl_ratio'] = self.strehl
                if config['dummy']:
                    self.AO_info['slope_AO_full']['strehl_ratio_2'] = self.strehl_2

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                if self.AO_settings['focus_enable'] == 0:
                    self.done.emit(4)
                elif self.focus_settings['focus_mode_flag'] == 0:
                    self.done2.emit(0)
                elif self.focus_settings['focus_mode_flag'] == 1:
                    self.done2.emit(1)

            # Finished closed-loop AO process
            if self.AO_settings['focus_enable'] == 0:
                self.done.emit(4)
            elif self.focus_settings['focus_mode_flag'] == 0:
                self.done2.emit(0)
            elif self.focus_settings['focus_mode_flag'] == 1:
                self.done2.emit(1)

        except Exception as e:
            raise
            self.error.emit(e)   

    @Slot(object)
    def stop(self):
        self.loop = False
        self.log = False