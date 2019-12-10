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
from HDF5_dset import make_dset, dset_append
from image_acquisition import acq_image
from centroid_acquisition import acq_centroid
from spot_sim import SpotSim

logger = log.get_logger(__name__)

class Calibration(QObject):
    """
    Calibrates deformable mirror and retrieves influence function + control matrix
    """
    start = Signal()
    write = Signal()
    done = Signal()
    error = Signal(object)
    image = Signal(object)
    message = Signal(object)
    cent = Signal(object)
    info = Signal(object)

    def __init__(self, sensor, mirror, settings):

        # Get search block settings
        self.SB_settings = settings

        # Get sensor instance
        self.sensor = sensor

        # Get mirror instance
        self.mirror = mirror

        # Initialise deformable mirror information parameter
        self.mirror_info = {}

        # Initialise influence function matrix
        self.inf_matrix_slopes = np.zeros([2 * self.SB_settings['act_ref_cent_num'], config['DM']['actuator_num']])
        
        super().__init__()

    @Slot(object)
    def run(self):
        try:
            # Set process flags
            self.calibrate = True
            self.calc_cent = True
            self.calc_inf = True
            self.log = True

            # Start thread
            self.start.emit()

            """
            Apply highest and lowest voltage to each actuator individually and retrieve raw slopes of each S-H spot

            Time for one calibration cycle for all actuators with image acquisition, but without centroiding: 56.686575999
            """
            # Initialise deformable mirror voltage array
            voltages = np.zeros(config['DM']['actuator_num'])
            
            prev1 = time.perf_counter()

            # Open HDF5 file and create new dataset to store calibration data
            data_set_img = np.zeros([self.SB_settings['sensor_width'], self.SB_settings['sensor_height']])
            data_set_cent = np.zeros(self.SB_settings['act_ref_cent_num'])
            self.output_file = h5py.File('data_info.h5', 'a')
            data_set = self.output_file['calibration_img']
            key_list_1 = ['dummy_calib_img', 'dummy_spot_cent_x', 'dummy_spot_cent_y']
            key_list_2 = ['real_calib_img']
            if config['dummy']:
                for k in key_list_1:
                    if k in data_set:
                        del data_set[k]
                    if k == 'dummy_calib_img':
                        make_dset(data_set, k, data_set_img)
                    elif k in {'dummy_spot_cent_x', 'dummy_spot_cent_y'}:
                        make_dset(data_set, k, data_set_cent)
            else:
                for k in key_list_2:
                    if k in data_set:
                        del data_set[k]
                    if k == 'real_calib_img':
                        make_dset(data_set, k, data_set_img)
            
            # Poke each actuator first in to vol_max, then to vol_min
            self.message.emit('DM calibration process started...')
            for i in range(config['DM']['actuator_num']):

                if self.calibrate:                    

                    try:
                        # print('On actuator', i + 1)

                        # Apply highest voltage
                        voltages[i] = config['DM']['vol_max']
                    
                        # Send values vector to mirror
                        self.mirror.Send(voltages)
                        
                        # Wait for DM to settle
                        time.sleep(config['DM']['settling_time'])
                        
                        # Acquire S-H spot image and display
                        if config['dummy']:
                            spot_img = SpotSim(self.SB_settings)
                            image_max, spot_cent_x, spot_cent_y = spot_img.SH_spot_sim(centred = 1)
                        else:
                            image_max = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)

                        # Image thresholding to remove background
                        image_max = image_max - config['image']['threshold'] * np.amax(image_max)
                        image_max[image_max < 0] = 0
                        self.image.emit(image_max)

                        # Append image to list
                        if config['dummy']:
                            dset_append(data_set, 'dummy_calib_img', image_max)
                            dset_append(data_set, 'dummy_spot_cent_x', spot_cent_x)
                            dset_append(data_set, 'dummy_spot_cent_y', spot_cent_y)
                        else:
                            dset_append(data_set, 'real_calib_img', image_max)

                        # Apply lowest voltage
                        voltages[i] = config['DM']['vol_min']

                        # Send values vector to mirror
                        self.mirror.Send(voltages)

                        # Wait for DM to settle
                        time.sleep(config['DM']['settling_time'])

                        # Acquire S-H spot image and display
                        if config['dummy']:
                            spot_img = SpotSim(self.SB_settings)
                            image_min, spot_cent_x, spot_cent_y = spot_img.SH_spot_sim(centred = 1)
                        else:
                            image_min = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)

                        # Image thresholding to remove background
                        image_min = image_min - config['image']['threshold'] * np.amax(image_min)
                        image_min[image_min < 0] = 0
                        self.image.emit(image_min)

                        # Append image to list
                        if config['dummy']:
                            dset_append(data_set, 'dummy_calib_img', image_min)
                            dset_append(data_set, 'dummy_spot_cent_x', spot_cent_x)
                            dset_append(data_set, 'dummy_spot_cent_y', spot_cent_y)
                        else:
                            dset_append(data_set, 'real_calib_img', image_min)

                        # Set actuator back to bias voltage
                        voltages[i] = config['DM']['vol_bias']
                    except Exception as e:
                        print(e)
                else:

                    self.done.emit()

            # Close HDF5 file
            self.output_file.close()

            prev2 = time.perf_counter()
            # print('Time for calibration image acquisition process is:', (prev2 - prev1))

            # Reset mirror
            self.mirror.Reset()

            # Calculate S-H spot centroids for each image in data list to get slopes
            if self.calc_cent:

                self.message.emit('Centroid calculation process started...')
                self.slope_x, self.slope_y = acq_centroid(self.SB_settings, flag = 1)
                self.message.emit('Centroid calculation process finished.')
            else:

                self.done.emit()

            # Fill influence function matrix with acquired slopes
            if self.calc_inf:
                
                for i in range(config['DM']['actuator_num']):

                    self.inf_matrix_slopes[:self.SB_settings['act_ref_cent_num'], i] = \
                        (self.slope_x[2 * i] - self.slope_x[2 * i + 1]) / (config['DM']['vol_max'] - config['DM']['vol_min'])
                    self.inf_matrix_slopes[self.SB_settings['act_ref_cent_num']:, i] = \
                        (self.slope_y[2 * i] - self.slope_y[2 * i + 1]) / (config['DM']['vol_max'] - config['DM']['vol_min'])             

                # print('Influence function is:', self.inf_matrix_slopes)

                # Calculate singular value decomposition of influence function matrix
                u, s, vh = np.linalg.svd(self.inf_matrix_slopes, full_matrices = False)

                # print('u: {}, s: {}, vh: {}'.format(u, s, vh))
                # print('The shapes of u, s, and vh are: {}, {}, and {}'.format(np.shape(u), np.shape(s), np.shape(vh)))

                # Calculate pseudo inverse of influence function matrix to get final control matrix
                self.control_matrix_slopes = np.linalg.pinv(self.inf_matrix_slopes)

                self.message.emit('DM calibration process finished.')
                # print('Control matrix is:', self.control_matrix_slopes)
                # print('Shape of control matrix is:', np.shape(self.control_matrix_slopes))
            else:

                self.done.emit()

            prev3 = time.perf_counter()
            print('Time for entire calibration process is:', (prev3 - prev1))      

            """
            Returns deformable mirror calibration information into self.mirror_info
            """ 
            if self.log:

                self.mirror_info['calib_slope_x'] = self.slope_x
                self.mirror_info['calib_slope_y'] = self.slope_y
                self.mirror_info['inf_matrix_slopes_SV'] = s
                self.mirror_info['inf_matrix_slopes'] = self.inf_matrix_slopes
                self.mirror_info['control_matrix_slopes'] = self.control_matrix_slopes

                self.info.emit(self.mirror_info)
                self.write.emit()
            else:

                self.done.emit()

            # Finished calibrating deformable mirror and retrieving influence functions
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)

    @Slot(object)
    def stop(self):
        self.calibrate = False
        self.calc_cent = False
        self.calc_inf = False
        self.log = False
