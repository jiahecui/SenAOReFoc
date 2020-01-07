from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import logging
import sys
import os
import argparse
import time
import h5py
import PIL.Image
import numpy as np

import log
from config import config
from HDF5_dset import dset_append, get_dset, get_mat_dset
from image_acquisition import acq_image
from centroid_acquisition import acq_centroid
from spot_sim import SpotSim
from gaussian_inf import inf
from common import get_slope_from_phase

logger = log.get_logger(__name__)

class AO_Zernikes(QObject):
    """
    Runs closed-loop AO using calibrated zernike control matrix
    """
    start = Signal()
    write = Signal()
    done = Signal(object)
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

        # Initialise zernike coefficient array
        self.zern_coeff = np.zeros([config['AO']['control_coeff_num'], 1])
        
        # Initialise array to record root mean square error and strehl ratio after each iteration
        self.loop_rms = np.zeros(config['AO']['loop_max'])
        self.strehl = np.zeros(config['AO']['loop_max'])

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
        phase_delta = (phase - phase_ave) * pupil_mask

        # print('Max and min values in phase before subtracting average: {}, {}'.format(np.amax(phase), np.amin(phase)))
        # print('Max and min values in phase after subtracting average: {}, {}'.format(np.amax(phase_delta), np.amin(phase_delta)))
        # print('phase_ave', phase_ave)

        # Calculate Strehl ratio estimated using only the statistics of the phase deviation, according to Mahajan
        phase_delta_2 = phase_delta ** 2 * pupil_mask
        sigma_2 = np.mean(phase_delta_2[pupil_mask])
        strehl = np.exp(-sigma_2)

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

        for i in range(config['DM']['actuator_num']):
            delta_phase += inf(coord_xx, coord_yy, self.mirror_settings['act_pos_x'], self.mirror_settings['act_pos_y'],\
                i, self.mirror_settings['act_diam']) * voltages[i]

        delta_phase = delta_phase * pupil_mask * 2

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
            (coord_yy * self.SB_settings['pixel_size']) ** 2) <= config['search_block']['pupil_diam'] * 1e3 / 2
        
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
            Normal closed-loop AO process WITH A FIXED GAIN, iterated until residual phase error is below value given by Marechel 
            criterion or iteration has reached maximum
            """
            # Initialise AO information parameter
            self.AO_info = {'zern_AO_1': {}}

            # Get pseudo-inverse of slope - zernike conversion matrix to translate zernike coefficients into slopes           
            conv_matrix_inv = np.linalg.pinv(self.mirror_settings['conv_matrix'])
        
            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'zern_AO_1', flag = 1)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['zern_AO_1']
            data_set_2 = data_file['AO_info']['zern_AO_1']

            self.message.emit('Process started for closed-loop AO via Zernikes...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(config['DM']['actuator_num'])

            prev1 = time.perf_counter()

            # Run closed-loop control until residual phase error is below a certain value or iteration has reached specified maximum
            for i in range(config['AO']['loop_max']):
                
                if self.loop:

                    try:
                        # Update mirror control voltages
                        if i == 0:
                            voltages = config['DM']['vol_bias']
                        else:
                            voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_zern'], \
                                zern_err[:config['AO']['control_coeff_num']]))

                            # print('Voltages {}: {}'.format(i, voltages))
                            print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

                        if config['dummy']:
                            if config['real_phase']:

                                # Update phase profile, slope data, S-H spot image, and calculate Strehl ratio 
                                if i == 0:

                                    # Retrieve phase profile
                                    phase_init = get_mat_dset(self.SB_settings, flag = 1)

                                    print('Max and min values of initial phase are: {} um, {} um'.format(np.amax(phase_init), np.amin(phase_init)))

                                    # Calculate strehl ratio of initial phase profile
                                    strehl_init = self.strehl_calc(phase_init / (config['AO']['lambda'] / (2 * np.pi)))
                                    self.strehl[i] = strehl_init

                                    print('Strehl ratio of initial phase is:', strehl_init)  

                                    # Retrieve slope data and S-H spot image from real phase profile
                                    slope_x_init, slope_y_init = get_mat_dset(self.SB_settings, flag = 2)
                                    spot_img = SpotSim(self.SB_settings)
                                    AO_image, act_cent_coord_x, act_cent_coord_y = spot_img.SH_spot_sim(centred = 1, xc = slope_x_init, yc = slope_y_init)

                                    # Convert 1d to 2d numpy array
                                    slope_x_init = np.reshape(slope_x_init, (1, len(slope_x_init)))
                                    slope_y_init = np.reshape(slope_y_init, (1, len(slope_y_init)))

                                    print('Max and min values of initial slope_x are: {}, {}'.format(np.amax(slope_x_init), np.amin(slope_x_init)))
                                    print('Max and min values of initial slope_y are: {}, {}'.format(np.amax(slope_y_init), np.amin(slope_y_init)))

                                    phase = phase_init.copy()
                                    strehl = strehl_init.copy()
                                    slope_x = slope_x_init.copy()
                                    slope_y = slope_y_init.copy()
                                else:

                                    # Calculate phase profile introduced by DM
                                    delta_phase = self.phase_calc(voltages)

                                    # Update phase data
                                    phase = phase_init - delta_phase

                                    print('Max and min values of phase are: {} um, {} um'.format(np.amax(phase), np.amin(phase)))

                                    # Calculate strehl ratio of updated phase profile
                                    strehl = self.strehl_calc(phase / (config['AO']['lambda'] / (2 * np.pi)))
                                    self.strehl[i] = strehl

                                    print('Strehl ratio of phase {} is: {}'.format(i + 1, strehl))

                                    # Retrieve slope data and S-H spot image from updated phase profile
                                    slope_x, slope_y = get_slope_from_phase(self.SB_settings, phase)
                                    spot_img = SpotSim(self.SB_settings)
                                    AO_image, act_cent_coord_x, act_cent_coord_y = spot_img.SH_spot_sim(centred = 1, xc = slope_x, yc = slope_y)

                                    print('Max and min values of slope_x are: {}, {}'.format(np.amax(slope_x), np.amin(slope_x)))
                                    print('Max and min values of slope_y are: {}, {}'.format(np.amax(slope_y), np.amin(slope_y)))

                                    # Convert 1d to 2d numpy array
                                    slope_x = np.reshape(slope_x, (1, len(slope_x)))
                                    slope_y = np.reshape(slope_y, (1, len(slope_y)))
                                    
                            else:

                                self.done.emit(1)
                        else:

                            # Send values vector to mirror
                            self.mirror.Send(voltages)
                            
                            # Wait for DM to settle
                            time.sleep(config['DM']['settling_time'])
                        
                            # Acquire S-H spots using camera and append to list
                            AO_image = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)
                            dset_append(data_set_1, 'real_AO_img', AO_image)

                            # Calculate centroids of S-H spots
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 3)
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                        # Image thresholding to remove background
                        AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                        AO_image[AO_image < 0] = 0
                        self.image.emit(AO_image)

                        if not config['dummy']:
                            # Draw actual S-H spot centroids on image layer
                            AO_image.ravel()[act_cent_coord.astype(int)] = 0
                            self.image.emit(AO_image)

                        # Concatenate slopes into one slope matrix
                        slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                        # Get detected zernike coefficients from slope matrix
                        self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                        # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                        zern_err = self.zern_coeff_detect.copy()
                        rms = np.sqrt((zern_err ** 2).mean())
                        self.loop_rms[i] = rms 

                        print('Root mean square error {} is {}'.format(i + 1, rms))                        

                        # Append data to list
                        if config['dummy']:
                            dset_append(data_set_2, 'dummy_spot_slope_x', slope_x)
                            dset_append(data_set_2, 'dummy_spot_slope_y', slope_y)
                            dset_append(data_set_2, 'dummy_spot_slope', slope)
                            dset_append(data_set_2, 'dummy_spot_zern_err', zern_err)
                        else:
                            dset_append(data_set_2, 'real_spot_slope_x', slope_x)
                            dset_append(data_set_2, 'real_spot_slope_y', slope_y)
                            dset_append(data_set_2, 'real_spot_slope', slope)
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

            self.message.emit('Process complete.')
            print('Final root mean square error of detected wavefront is: {} microns'.format(rms))

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is:', (prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['zern_AO_1']['loop_num'] = i + 1
                self.AO_info['zern_AO_1']['residual_phase_err_1'] = self.loop_rms
                self.AO_info['zern_AO_1']['strehl_ratio'] = self.strehl

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit(1)
            try:
                # Finished closed-loop AO process
                self.done.emit(1)
            except Exception as e:
                print(e)

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
            Closed-loop AO process to handle obscured S-H spots using a FIXED GAIN, iterated until residual phase error is below value 
            given by Marechel criterion or iteration has reached maximum
            """ 
            # Initialise AO information parameter
            self.AO_info = {'zern_AO_2': {}}

            # Get pseudo-inverse of slope - zernike conversion matrix to translate zernike coefficients into slopes           
            conv_matrix_inv = np.linalg.pinv(self.mirror_settings['conv_matrix'])
        
            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'zern_AO_2', flag = 1)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['zern_AO_2']
            data_set_2 = data_file['AO_info']['zern_AO_2']

            self.message.emit('Process started for closed-loop AO via Zernikes with obscured subapertures...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(config['DM']['actuator_num'])

            prev1 = time.perf_counter()

            # Run closed-loop control until residual phase error is below a certain value or iteration has reached specified maximum
            for i in range(config['AO']['loop_max']):
                
                if self.loop:

                    try:
                        if config['dummy']:

                            pass
                        else:

                            # Update mirror control voltages
                            if i == 0:
                                voltages = config['DM']['vol_bias']
                            else:
                                voltages -= config['AO']['loop_gain'] * np.dot(control_matrix_zern, zern_err)                        

                            # Send values vector to mirror
                            self.mirror.Send(voltages)
                            
                            # Wait for DM to settle
                            time.sleep(config['DM']['settling_time'])
                        
                            # Acquire S-H spots using camera and append to list
                            AO_image = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)
                            dset_append(data_set_1, 'real_AO_img', AO_image)

                        # Image thresholding to remove background
                        AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                        AO_image[AO_image < 0] = 0
                        self.image.emit(AO_image)

                        # Calculate centroids of S-H spots
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 5)
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                        # Remove corresponding elements from slopes and rows from influence function matrix, zernike matrix and zernike derivative matrix
                        index_remove = np.where(slope_x + self.mirror_settings['act_ref_cent_coord_x'] == 0)[0]
                        index_remove_inf = np.concatenate((index_remove, index_remove + self.SB_settings['act_ref_cent_num']), axis = None)
                        slope_x = np.argwhere(slope_x + self.mirror_settings['act_ref_cent_coord_x'])
                        slope_y = np.argwhere(slope_y + self.mirror_settings['act_ref_cent_coord_y'])
                        act_cent_coord = np.delete(act_cent_coord, index_remove, axis = None)
                        zern_matrix = np.delete(self.mirror_settings['zern_matrix'].copy(), index_remove, axis = 0)
                        inf_matrix_slopes = np.delete(self.mirror_settings['inf_matrix_slopes'].copy(), index_remove_inf, axis = 0)
                        diff_matrix = np.delete(self.mirror_settings['diff_matrix'].copy(), index_remove_inf, axis = 0)

                        # print('The number of obscured subapertures is: {}'.format(len(index_remove)))
                        # print('The shapes of slope_x, slope_y, and inf_matrix_slopes are: {}, {}, and {}'.format(np.shape(slope_x), np.shape(slope_y),\
                        #     np.shape(inf_matrix_slopes)))

                        # Draw actual S-H spot centroids on image layer
                        AO_image.ravel()[act_cent_coord.astype(int)] = 0
                        self.image.emit(AO_image)

                        # Recalculate Cholesky decomposition of np.dot(zern_matrix.T, zern_matrix)
                        p_matrix = np.linalg.cholesky(np.dot(zern_matrix.T, zern_matrix))

                        # Recalculate conversion matrix
                        conv_matrix = np.dot(p_matrix, np.linalg.pinv(diff_matrix))

                        # Recalculate control function via zernikes
                        control_matrix_zern = np.linalg.pinv(np.dot(conv_matrix, inf_matrix_slopes))

                        # Concatenate slopes into one slope matrix
                        slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                        # Get detected zernike coefficients from slope matrix
                        self.zern_coeff_detect = np.dot(conv_matrix, slope)

                        # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                        zern_err = self.zern_coeff_detect.copy()
                        rms = np.sqrt((zern_err ** 2).mean(axis = 0))[0]
                        self.loop_rms[i] = rms

                        print('Root mean square error {} is {}'.format(i + 1, rms))                        

                        # Append data to list
                        if config['dummy']:
                            dset_append(data_set_2, 'dummy_spot_slope_x', slope_x)
                            dset_append(data_set_2, 'dummy_spot_slope_y', slope_y)
                            dset_append(data_set_2, 'dummy_spot_slope', slope)
                            dset_append(data_set_2, 'dummy_spot_zern_err', zern_err)
                        else:
                            dset_append(data_set_2, 'real_spot_slope_x', slope_x)
                            dset_append(data_set_2, 'real_spot_slope_y', slope_y)
                            dset_append(data_set_2, 'real_spot_slope', slope)
                            dset_append(data_set_2, 'real_spot_zern_err', zern_err)

                        # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                        if rms <= config['AO']['tolerance_fact_zern']:
                            break                 

                    except Exception as e:
                        print(e)
                else:

                    self.done.emit(2)

            # Close HDF5 file
            data_file.close()

            self.message.emit('Process complete.')
            print('Final root mean square error of detected wavefront is: {} microns'.format(rms))

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is:', (prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['zern_AO_2']['loop_num'] = i
                self.AO_info['zern_AO_2']['residual_phase_err_1'] = self.loop_rms

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
            Closed-loop AO process to handle partial correction using a FIXED GAIN, iterated until residual phase error is below value 
            given by Marechel criterion or iteration has reached maximum            
            """
            # Initialise AO information parameter
            self.AO_info = {'zern_AO_3': {}}

            # Get pseudo-inverse of slope - zernike conversion matrix to translate zernike coefficients into slopes           
            conv_matrix_inv = np.linalg.pinv(self.mirror_settings['conv_matrix'])
        
            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'zern_AO_3', flag = 1)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['zern_AO_3']
            data_set_2 = data_file['AO_info']['zern_AO_3']

            self.message.emit('Process started for closed-loop AO via Zernikes with partial correction...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(config['DM']['actuator_num'])

            prev1 = time.perf_counter()

            # Run closed-loop control until residual phase error is below a certain value or iteration has reached specified maximum
            for i in range(config['AO']['loop_max']):
                
                if self.loop:

                    try:
                        if config['dummy']:

                            pass
                        else:

                            # Update mirror control voltages
                            if i == 0:
                                voltages = config['DM']['vol_bias']
                            else:
                                zern_err[0 : 2, 0] = 0
                                voltages -= config['AO']['loop_gain'] * np.dot(self.mirror_settings['control_matrix_zern'], zern_err)                        

                            # Send values vector to mirror
                            self.mirror.Send(voltages)
                            
                            # Wait for DM to settle
                            time.sleep(config['DM']['settling_time'])
                        
                            # Acquire S-H spots using camera and append to list
                            AO_image = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)
                            dset_append(data_set_1, 'real_AO_img', AO_image)

                        # Image thresholding to remove background
                        AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                        AO_image[AO_image < 0] = 0
                        self.image.emit(AO_image)

                        # Calculate centroids of S-H spots
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 7)
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                        # Draw actual S-H spot centroids on image layer
                        AO_image.ravel()[act_cent_coord.astype(int)] = 0
                        self.image.emit(AO_image)

                        # Concatenate slopes into one slope matrix
                        slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                        # Get detected zernike coefficients from slope matrix
                        self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                        # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                        zern_err = self.zern_coeff_detect.copy()
                        rms = np.sqrt((zern_err ** 2).mean(axis = 0))[0]
                        self.loop_rms[i] = rms

                        print('Root mean square error {} is {}'.format(i + 1, rms))                        

                        # Append data to list
                        if config['dummy']:
                            dset_append(data_set_2, 'dummy_spot_slope_x', slope_x)
                            dset_append(data_set_2, 'dummy_spot_slope_y', slope_y)
                            dset_append(data_set_2, 'dummy_spot_slope', slope)
                            dset_append(data_set_2, 'dummy_spot_zern_err', zern_err)
                        else:
                            dset_append(data_set_2, 'real_spot_slope_x', slope_x)
                            dset_append(data_set_2, 'real_spot_slope_y', slope_y)
                            dset_append(data_set_2, 'real_spot_slope', slope)
                            dset_append(data_set_2, 'real_spot_zern_err', zern_err)

                        # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                        if rms <= config['AO']['tolerance_fact_zern']:
                            break                 

                    except Exception as e:
                        print(e)
                else:

                    self.done.emit(3)

            # Close HDF5 file
            data_file.close()

            self.message.emit('Process complete.')
            print('Final root mean square error of detected wavefront is: {} microns'.format(rms))

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is:', (prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['zern_AO_3']['loop_num'] = i
                self.AO_info['zern_AO_3']['residual_phase_err_1'] = self.loop_rms

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit(3)

            # Finished closed-loop AO process
            self.done.emit(3)

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
            Closed-loop AO process to handle both obscured S-H spots and partial correction using a FIXED GAIN, iterated until residual phase 
            error is below value given by Marechel criterion or iteration has reached maximum
            """
            # Initialise AO information parameter
            self.AO_info = {'zern_AO_full': {}}

            # Get pseudo-inverse of slope - zernike conversion matrix to translate zernike coefficients into slopes           
            conv_matrix_inv = np.linalg.pinv(self.mirror_settings['conv_matrix'])

            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'zern_AO_full', flag = 1)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['zern_AO_full']
            data_set_2 = data_file['AO_info']['zern_AO_full']

            self.message.emit('Process started for full closed-loop AO via Zernikes...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(config['DM']['actuator_num'])

            prev1 = time.perf_counter()

            # Run closed-loop control until residual phase error is below a certain value or iteration has reached specified maximum
            for i in range(config['AO']['loop_max']):
                
                if self.loop:

                    try:
                        if config['dummy']:

                            pass
                        else:

                            # Update mirror control voltages
                            if i == 0:
                                voltages = config['DM']['vol_bias']
                            else:
                                zern_err[0 : 2, 0] = 0
                                voltages -= config['AO']['loop_gain'] * np.dot(control_matrix_zern, zern_err)                        

                            # Send values vector to mirror
                            self.mirror.Send(voltages)
                            
                            # Wait for DM to settle
                            time.sleep(config['DM']['settling_time'])
                        
                            # Acquire S-H spots using camera and append to list
                            AO_image = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)
                            dset_append(data_set_1, 'real_AO_img', AO_image)

                        # Image thresholding to remove background
                        AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                        AO_image[AO_image < 0] = 0
                        self.image.emit(AO_image)

                        # Calculate centroids of S-H spots
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 9)
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                        # Remove corresponding elements from slopes and rows from influence function matrix, zernike matrix and zernike derivative matrix
                        index_remove = np.where(slope_x + self.mirror_settings['act_ref_cent_coord_x'] == 0)[0]
                        index_remove_inf = np.concatenate((index_remove, index_remove + self.SB_settings['act_ref_cent_num']), axis = None)
                        slope_x = np.argwhere(slope_x + self.mirror_settings['act_ref_cent_coord_x'])
                        slope_y = np.argwhere(slope_y + self.mirror_settings['act_ref_cent_coord_y'])
                        act_cent_coord = np.delete(act_cent_coord, index_remove, axis = None)
                        zern_matrix = np.delete(self.mirror_settings['zern_matrix'].copy(), index_remove, axis = 0)
                        inf_matrix_slopes = np.delete(self.mirror_settings['inf_matrix_slopes'].copy(), index_remove_inf, axis = 0)
                        diff_matrix = np.delete(self.mirror_settings['diff_matrix'].copy(), index_remove_inf, axis = 0)

                        # print('The number of obscured subapertures is: {}'.format(len(index_remove)))
                        # print('The shapes of slope_x, slope_y, and inf_matrix_slopes are: {}, {}, and {}'.format(np.shape(slope_x), np.shape(slope_y),\
                        #     np.shape(inf_matrix_slopes)))

                        # Draw actual S-H spot centroids on image layer
                        AO_image.ravel()[act_cent_coord.astype(int)] = 0
                        self.image.emit(AO_image)

                        # Recalculate Cholesky decomposition of np.dot(zern_matrix.T, zern_matrix)
                        p_matrix = np.linalg.cholesky(np.dot(zern_matrix.T, zern_matrix))

                        # Recalculate conversion matrix
                        conv_matrix = np.dot(p_matrix, np.linalg.pinv(diff_matrix))

                        # Recalculate control function via zernikes
                        control_matrix_zern = np.linalg.pinv(np.dot(conv_matrix, inf_matrix_slopes))

                        # Concatenate slopes into one slope matrix
                        slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                        # Get detected zernike coefficients from slope matrix
                        self.zern_coeff_detect = np.dot(conv_matrix, slope)

                        # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                        zern_err = self.zern_coeff_detect.copy()
                        rms = np.sqrt((zern_err ** 2).mean(axis = 0))[0]
                        self.loop_rms[i] = rms

                        print('Root mean square error {} is {}'.format(i + 1, rms))                        

                        # Append data to list
                        if config['dummy']:
                            dset_append(data_set_2, 'dummy_spot_slope_x', slope_x)
                            dset_append(data_set_2, 'dummy_spot_slope_y', slope_y)
                            dset_append(data_set_2, 'dummy_spot_slope', slope)
                            dset_append(data_set_2, 'dummy_spot_zern_err', zern_err)
                        else:
                            dset_append(data_set_2, 'real_spot_slope_x', slope_x)
                            dset_append(data_set_2, 'real_spot_slope_y', slope_y)
                            dset_append(data_set_2, 'real_spot_slope', slope)
                            dset_append(data_set_2, 'real_spot_zern_err', zern_err)

                        # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                        if rms <= config['AO']['tolerance_fact_zern']:
                            break                 

                    except Exception as e:
                        print(e)
                else:

                    self.done.emit(4)

            # Close HDF5 file
            data_file.close()

            self.message.emit('Process complete.')
            print('Final root mean square error of detected wavefront is: {} microns'.format(rms))

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is:', (prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['zern_AO_full']['loop_num'] = i
                self.AO_info['zern_AO_full']['residual_phase_err_1'] = self.loop_rms

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit(4)

            # Finished closed-loop AO process
            self.done.emit(4)

        except Exception as e:
            raise
            self.error.emit(e)

    @Slot(object)
    def stop(self):
        self.loop = False
        self.log = False

