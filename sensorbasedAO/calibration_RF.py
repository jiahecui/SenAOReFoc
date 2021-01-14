from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import sys
import os
import argparse
import time
import click
import h5py
import numpy as np

import log
from config import config
from HDF5_dset import dset_append, get_dset
from image_acquisition import acq_image
from centroid_acquisition import acq_centroid

logger = log.get_logger(__name__)

class Calibration_RF(QObject):
    """
    Calibrates remote focusing using nulling correction 
    """
    start = Signal()
    write = Signal()
    calib_write = Signal()
    done = Signal()
    error = Signal(object)
    image = Signal(object)
    message = Signal(object)
    info = Signal(object)
    calib_info = Signal(object)

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

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_0']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_1']

        # Initialise deformable mirror information parameter
        self.mirror_info = {}

        # Inilialise remote focusing calibration AO information parameter
        self.calib_RF_AO_info = {}

        # Initialise array to store remote focusing calibration voltages
        self.calib_array = np.zeros([self.actuator_num, config['RF_calib']['calib_step_num']])

        # Initialise array to store final value of Zernike modes / RMS Zernike value / Strehl ratio for each calibration step
        self.calib_slope_x = np.zeros([config['RF_calib']['calib_step_num'] * 2, self.SB_settings['act_ref_cent_num']])
        self.calib_slope_y = np.zeros([config['RF_calib']['calib_step_num'] * 2, self.SB_settings['act_ref_cent_num']])
        self.calib_zern_coeff = np.zeros([config['RF_calib']['calib_step_num'] * 2, config['AO']['control_coeff_num']])
        self.calib_rms_zern = np.zeros([config['RF_calib']['calib_step_num'], 2])
        self.calib_strehl = np.zeros([config['RF_calib']['calib_step_num'], 2])

        super().__init__()

    @Slot(object)
    def run0(self):
        try:
            # Set process flags
            self.loop = True
            self.log = True

            # Start thread
            self.start.emit()

            """
            Calibrates for remote focusing by moving a card sample to different positions along axis,
            correcting for that amount of displacement, then storing the actuator voltages for each position
            """
            # Create new datasets in HDF5 file to store remote focusing calibration data
            get_dset(self.SB_settings, 'calibration_RF_img', flag = 5)
            data_file = h5py.File('data_info.h5', 'a')
            data_set = data_file['calibration_RF_img']

            self.message.emit('\nProcess started for calibration of remote focusing in negative direction...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            # Iterate through each step in positive increments and correct for the introduced displacement
            for l in range(config['RF_calib']['calib_step_num']):

                # Ask user to move sample to different positions along the z-axis
                self.message.emit('\nMove sample to negative position {}. \nPress [y] to confirm. \nPress [s] to save and exit.\
                    \nPress [d] to discard and exit.'.format(l + 1))
                c = click.getchar()

                while True:
                    if c == 'y':
                        break
                    elif c == 's':
                        data_file.close()
                        self.mirror.Reset()
                        self.mirror_info['remote_focus_neg_voltages'] = self.calib_array
                        self.calib_RF_AO_info['calib_RF_neg_slope_x'] = self.calib_slope_x
                        self.calib_RF_AO_info['calib_RF_neg_slope_y'] = self.calib_slope_y
                        self.calib_RF_AO_info['calib_RF_neg_zern_coeff'] = self.calib_zern_coeff
                        self.calib_RF_AO_info['calib_RF_neg_rms_zern'] = self.calib_rms_zern
                        self.calib_RF_AO_info['calib_RF_neg_strehl'] = self.calib_strehl
                        self.info.emit(self.mirror_info)
                        self.calib_info.emit(self.calib_RF_AO_info)
                        self.write.emit()
                        self.calib_write.emit()
                        self.message.emit('\nCalibration saved. Exit.')
                        self.done.emit()
                    elif c == 'd':
                        data_file.close()
                        self.mirror.Reset()
                        self.message.emit('\nCalibration discarded. Exit.')
                        self.done.emit()
                    else:
                        self.message.emit('\nInvalid input. Please try again.')

                    c = click.getchar()

                if self.loop:

                    # Run closed-loop control until tolerance value or maximum loop iteration is reached
                    for i in range(self.AO_settings['loop_max'] + 1):
                    
                        try:

                            # Update mirror control voltages
                            if l == 0 and i == 0:                     
                                voltages[:] = config['DM']['vol_bias']
                            elif l > 0 and i == 0:
                                voltages = self.calib_array[:, l - 1].copy()
                            elif i > 0:
                                voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_slopes'], slope_err))

                            # voltages[voltages < -1.0] = -1.0
                            # voltages[voltages > 1.0] = 1.0
                            
                            print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

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
                            dset_append(data_set, 'real_calib_RF_img', AO_image)

                            # Calculate centroids of S-H spots
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 11)
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])
                        
                            # print('slope_x:', slope_x)
                            # print('slope_y:', slope_y)

                            # Draw actual S-H spot centroids on image layer
                            AO_image.ravel()[act_cent_coord.astype(int)] = 0
                            self.image.emit(AO_image)

                            # Take tip\tilt off
                            slope_x -= np.mean(slope_x)
                            slope_y -= np.mean(slope_y)

                            # Concatenate slopes into one slope matrix
                            slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                            # Get residual slope error and calculate root mean square (rms) error
                            slope_err = slope.copy()

                            # Get detected zernike coefficients from slope matrix
                            self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                            # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                            zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                            zern_err_part[[0, 1], 0] = 0
                            rms_zern = np.sqrt((zern_err ** 2).sum())
                            rms_zern_part = np.sqrt((zern_err_part ** 2).sum())

                            strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)

                            print('Strehl ratio {} from rms_zern_part is: {}'.format(i, strehl))

                            if i == 0:
                                self.calib_slope_x[2 * l, :] = slope_x
                                self.calib_slope_y[2 * l, :] = slope_y
                                self.calib_zern_coeff[2 * l, :] = self.zern_coeff_detect[:config['AO']['control_coeff_num'], 0].T
                                self.calib_rms_zern[l, 0] = rms_zern_part
                                self.calib_strehl[l, 0] = strehl

                            if strehl >= config['AO']['tolerance_fact_strehl'] or i == self.AO_settings['loop_max']:
                                self.calib_slope_x[2 * l + 1, :] = slope_x
                                self.calib_slope_y[2 * l + 1, :] = slope_y
                                self.calib_zern_coeff[2 * l + 1, :] = self.zern_coeff_detect[:config['AO']['control_coeff_num'], 0].T
                                self.calib_rms_zern[l, 1] = rms_zern_part
                                self.calib_strehl[l, 1] = strehl
                                self.calib_array[:, l] = voltages
                                break

                        except Exception as e:
                            print(e)

                else:

                    self.done.emit()

            # Close HDF5 file
            data_file.close()

            self.message.emit('\nProcess complete.')

            # Reset mirror
            self.mirror.Reset()

            """
            Returns remote focusing calibration information into self.mirror_info, self.calib_RF_AO_info
            """ 
            if self.log:

                self.mirror_info['remote_focus_neg_voltages'] = self.calib_array
                self.calib_RF_AO_info['calib_RF_neg_slope_x'] = self.calib_slope_x
                self.calib_RF_AO_info['calib_RF_neg_slope_y'] = self.calib_slope_y
                self.calib_RF_AO_info['calib_RF_neg_zern_coeff'] = self.calib_zern_coeff
                self.calib_RF_AO_info['calib_RF_neg_rms_zern'] = self.calib_rms_zern
                self.calib_RF_AO_info['calib_RF_neg_strehl'] = self.calib_strehl

                self.info.emit(self.mirror_info)
                self.calib_info.emit(self.calib_RF_AO_info)
                self.write.emit()
                self.calib_write.emit()
            else:

                self.done.emit()

            # Finished remote focusing calibration process
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
            Calibrates for remote focusing by moving a card sample to different positive positions along axis,
            correcting for that amount of displacement, then storing the actuator voltages for each position
            """
            # Create new datasets in HDF5 file to store remote focusing calibration data
            get_dset(self.SB_settings, 'calibration_RF_img', flag = 5)
            data_file = h5py.File('data_info.h5', 'a')
            data_set = data_file['calibration_RF_img']

            self.message.emit('\nProcess started for calibration of remote focusing in positive direction...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            # Iterate through each step in positive increments and correct for the introduced displacement
            for l in range(config['RF_calib']['calib_step_num']):

                # Ask user to move sample to different positions along the z-axis
                self.message.emit('\nMove sample to positive position {}. \nPress [y] to confirm. \nPress [s] to save and exit.\
                    \nPress [d] to discard and exit.'.format(l + 1))
                c = click.getchar()

                while True:
                    if c == 'y':
                        break
                    elif c == 's':
                        data_file.close()
                        self.mirror.Reset()
                        self.mirror_info['remote_focus_pos_voltages'] = self.calib_array
                        self.calib_RF_AO_info['calib_RF_pos_slope_x'] = self.calib_slope_x
                        self.calib_RF_AO_info['calib_RF_pos_slope_y'] = self.calib_slope_y
                        self.calib_RF_AO_info['calib_RF_pos_zern_coeff'] = self.calib_zern_coeff
                        self.calib_RF_AO_info['calib_RF_pos_rms_zern'] = self.calib_rms_zern
                        self.calib_RF_AO_info['calib_RF_pos_strehl'] = self.calib_strehl
                        self.info.emit(self.mirror_info)
                        self.calib_info.emit(self.calib_RF_AO_info)
                        self.write.emit()
                        self.calib_write.emit()
                        self.message.emit('\nCalibration saved. Exit.')
                        self.done.emit()
                    elif c == 'd':
                        data_file.close()
                        self.mirror.Reset()
                        self.message.emit('\nCalibration discarded. Exit.')
                        self.done.emit()
                    else:
                        self.message.emit('\nInvalid input. Please try again.')

                    c = click.getchar()

                if self.loop:

                    # Run closed-loop control until tolerance value or maximum loop iteration is reached
                    for i in range(self.AO_settings['loop_max'] + 1):
                    
                        try:

                            # Update mirror control voltages
                            if l == 0 and i == 0:                     
                                voltages[:] = config['DM']['vol_bias']
                            elif l > 0 and i == 0:
                                voltages = self.calib_array[:, l - 1].copy()
                            elif i > 0:
                                voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_slopes'], slope_err))

                            # voltages[voltages < -1.0] = -1.0
                            # voltages[voltages > 1.0] = 1.0

                            print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

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
                            dset_append(data_set, 'real_calib_RF_img', AO_image)

                            # Calculate centroids of S-H spots
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 11)
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])
                        
                            # print('slope_x:', slope_x)
                            # print('slope_y:', slope_y)

                            # Draw actual S-H spot centroids on image layer
                            AO_image.ravel()[act_cent_coord.astype(int)] = 0
                            self.image.emit(AO_image)

                            # Take tip\tilt off
                            slope_x -= np.mean(slope_x)
                            slope_y -= np.mean(slope_y)

                            # Concatenate slopes into one slope matrix
                            slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                            # Get residual slope error and calculate root mean square (rms) error
                            slope_err = slope.copy()

                            # Get detected zernike coefficients from slope matrix
                            self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                            # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                            zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                            zern_err_part[[0, 1], 0] = 0
                            rms_zern = np.sqrt((zern_err ** 2).sum())
                            rms_zern_part = np.sqrt((zern_err_part ** 2).sum())

                            strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)

                            print('Strehl ratio {} from rms_zern_part is: {}'.format(i, strehl))

                            if i == 0:
                                self.calib_slope_x[2 * l, :] = slope_x
                                self.calib_slope_y[2 * l, :] = slope_y
                                self.calib_zern_coeff[2 * l, :] = self.zern_coeff_detect[:config['AO']['control_coeff_num'], 0].T
                                self.calib_rms_zern[l, 0] = rms_zern_part
                                self.calib_strehl[l, 0] = strehl

                            if strehl >= config['AO']['tolerance_fact_strehl'] or i == self.AO_settings['loop_max']:
                                self.calib_slope_x[2 * l + 1, :] = slope_x
                                self.calib_slope_y[2 * l + 1, :] = slope_y
                                self.calib_zern_coeff[2 * l + 1, :] = self.zern_coeff_detect[:config['AO']['control_coeff_num'], 0].T
                                self.calib_rms_zern[l, 1] = rms_zern_part
                                self.calib_strehl[l, 1] = strehl
                                self.calib_array[:, l] = voltages
                                break

                        except Exception as e:
                            print(e)

                else:

                    self.done.emit()

            # Close HDF5 file
            data_file.close()

            self.message.emit('\nProcess complete.')

            # Reset mirror
            self.mirror.Reset()

            """
            Returns remote focusing calibration information into self.mirror_info, self.calib_RF_AO_info
            """ 
            if self.log:

                self.mirror_info['remote_focus_pos_voltages'] = self.calib_array
                self.calib_RF_AO_info['calib_RF_pos_slope_x'] = self.calib_slope_x
                self.calib_RF_AO_info['calib_RF_pos_slope_y'] = self.calib_slope_y
                self.calib_RF_AO_info['calib_RF_pos_zern_coeff'] = self.calib_zern_coeff
                self.calib_RF_AO_info['calib_RF_pos_rms_zern'] = self.calib_rms_zern
                self.calib_RF_AO_info['calib_RF_pos_strehl'] = self.calib_strehl

                self.info.emit(self.mirror_info)
                self.calib_info.emit(self.calib_RF_AO_info)
                self.write.emit()
                self.calib_write.emit()
            else:

                self.done.emit()

            # Finished remote focusing calibration process
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)

    @Slot(object)
    def stop(self):
        self.loop = False
        self.log = False
