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
    done = Signal()
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

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_0']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_1']

        # Initialise deformable mirror information parameter
        self.mirror_info = {}

        # Initialise array to store remote focusing calibration voltages
        self.calib_array = np.zeros([self.actuator_num, config['RF_calib']['step_num']])

        super().__init__()

    @Slot(object)
    def run(self):
        try:
            # Set process flags
            self.loop = True
            self.log = True

            # Start thread
            self.start.emit()

            """
            Calibrates for remote focusing by moving a mirror sample to different positions along axis,
            correcting for that amount of defocus, then storing the actuator voltages for each position
            """
            # Create new datasets in HDF5 file to store remote focusing calibration data
            get_dset(self.SB_settings, 'calibration_RF_img', flag = 5)
            data_file = h5py.File('data_info.h5', 'a')
            data_set = data_file['calibration_RF_img']

            self.message.emit('\nProcess started for calibration of remote focusing...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            prev1 = time.perf_counter()

            # Iterate through each position and retrieve voltages to correct for the defocus at that position
            for l in range(config['RF_calib']['step_num']):

                # Ask user to move mirror sample to different positions along the z-axis
                self.message.emit('\nMove mirror to position {}. Press [y] to confirm.'.format(l))
                c = click.getchar()

                while True:
                    if c == 'y':
                        break
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

                            if strehl >= config['AO']['tolerance_fact_strehl'] or i == self.AO_settings['loop_max']:
                                self.calib_array[:, l] = voltages
                                break

                        except Exception as e:
                            print(e)

                else:

                    self.done.emit()

            # Close HDF5 file
            data_file.close()

            self.message.emit('\nProcess complete.')

            prev2 = time.perf_counter()
            print('Time for remote focusing calibration process is:', (prev2 - prev1))

            # Reset mirror
            self.mirror.Reset()

            """
            Returns remote focusing calibration information into self.mirror_info
            """ 
            if self.log:

                self.mirror_info['remote_focus_voltages'] = self.calib_array

                self.info.emit(self.mirror_info)
                self.write.emit()
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
