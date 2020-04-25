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
from HDF5_dset import dset_append, get_dset
from image_acquisition import acq_image
from centroid_acquisition import acq_centroid
from spot_sim import SpotSim
from gaussian_inf import inf_diff

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
    info = Signal(object)

    def __init__(self, sensor, mirror, settings):

        # Get search block settings
        self.SB_settings = settings

        # Get sensor instance
        self.sensor = sensor

        # Get mirror instance
        self.mirror = mirror

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
            self.pitch = config['DM0']['pitch']
            self.aperture = config['DM0']['aperture']
            self.pupil_diam = config['search_block']['pupil_diam_0']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']
            self.pitch = config['DM1']['pitch']
            self.aperture = config['DM1']['aperture']
            self.pupil_diam = config['search_block']['pupil_diam_1']

        # Initialise deformable mirror information parameter
        self.mirror_info = {}

        # Initialise influence function matrix
        self.inf_matrix_slopes = np.zeros([2 * self.SB_settings['act_ref_cent_num'], self.actuator_num])
        
        super().__init__()

    def act_coord_1(self, act_diam):
        """
        Calculates actuator position coordinates according to DM geometry (Alpao69)
        """
        xc, yc = (np.zeros(self.actuator_num) for i in range(2))

        for i in range(5):

            xc[i] = -4 * act_diam
            xc[self.actuator_num - 1 - i] = 4 * act_diam
            yc[i] = (2 - i) * act_diam
            yc[self.actuator_num - 1 - i] = (-2 + i) * act_diam

        for i in range(7):

            xc[5 + i] = -3 * act_diam
            xc[self.actuator_num - 6 - i] = 3 * act_diam
            yc[5 + i] = (3 - i) * act_diam
            yc[self.actuator_num - 6 - i] = (-3 + i) * act_diam

        for i in range(9):

            xc[12 + i] = -2 * act_diam
            xc[21 + i] = -act_diam
            xc[30 + i] = 0
            xc[self.actuator_num - 13 - i] = 2 * act_diam
            xc[self.actuator_num - 22 - i] = act_diam
            yc[12 + i] = (4 - i) * act_diam
            yc[21 + i] = (4 - i) * act_diam
            yc[30 + i] = (4 - i) * act_diam
            yc[self.actuator_num - 13 - i] = (-4 + i) * act_diam
            yc[self.actuator_num - 22 - i] = (-4 + i) * act_diam

        return xc, yc

    def act_coord_2(self, act_diam):
        """
        Calculates actuator position coordinates according to DM geometry (Boston140)
        """
        xc, yc = (np.zeros(self.actuator_num) for i in range(2))

        for i in range(10):

            xc[i] = (-5 - 0.5) * act_diam
            xc[self.actuator_num - 1 - i] = (5 + 0.5) * act_diam
            yc[i] = (4 + 0.5 - i) * act_diam
            yc[self.actuator_num - 1 - i] = (-4 - 0.5 + i) * act_diam

        for i in range(120):

            if i in [11,23,35,47,59] or i > 59:
                xc[10 + i] = (int(((10 + i) - self.actuator_num // 2) // 12) + 0.5) * act_diam
            else:
                xc[10 + i] = (int(((10 + i) - (self.actuator_num // 2 - 1)) // 12) + 0.5) * act_diam

            yc[10 + i] = (-(i % 12) + (5 + 0.5)) * act_diam

        return xc, yc

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

            if config['dummy']:

                """
                Get DM control matrix via slopes by modeling influence function of each actuator as a Gaussian function
                """
                # Get diameter spacing of one actuator
                act_diam = self.pupil_diam / self.aperture * self.pitch / self.SB_settings['pixel_size']

                # Get actuator coordinates
                if config['DM']['DM_num'] == 0:
                    xc, yc = self.act_coord_1(act_diam)
                elif config['DM']['DM_num'] == 1:
                    xc, yc = self.act_coord_2(act_diam)

                # Get size of individual elements within each search block
                self.elem_size = self.SB_settings['SB_diam'] / config['search_block']['div_elem']

                # Get reference centroid coordinates with pupil centre as zero coordinate
                if (self.SB_settings['SB_across_width'] % 2 == 0 and self.SB_settings['sensor_width'] % 2 == 0) or \
                    (self.SB_settings['SB_across_width'] % 2 == 1 and self.SB_settings['sensor_width'] % 2 == 1):

                    self.centred_ref_cent_coord_x = self.SB_settings['act_ref_cent_coord_x'] - self.SB_settings['sensor_width'] // 2
                    self.centred_ref_cent_coord_y = self.SB_settings['act_ref_cent_coord_y'] - self.SB_settings['sensor_width'] // 2

                else:

                    self.centred_ref_cent_coord_x = self.SB_settings['act_ref_cent_coord_x'] - (self.SB_settings['sensor_width'] // 2 - self.SB_settings['SB_rad'])
                    self.centred_ref_cent_coord_y = self.SB_settings['act_ref_cent_coord_y'] - (self.SB_settings['sensor_width'] // 2 - self.SB_settings['SB_rad'])

                # Start calibrating DM by calculating averaged derivatives of Gaussian distribution actuator influence function
                if self.calibrate:

                    self.message.emit('DM calibration process started...')
                    for i in range(self.SB_settings['act_ref_cent_num']):

                        # Get reference centroid coords of each element
                        elem_ref_cent_coord_x = np.arange(self.centred_ref_cent_coord_x[i] - self.SB_settings['SB_rad'] + self.elem_size / 2, \
                            self.centred_ref_cent_coord_x[i] + self.SB_settings['SB_rad'] - self.elem_size / 2, self.elem_size)
                        elem_ref_cent_coord_y = np.arange(self.centred_ref_cent_coord_y[i] - self.SB_settings['SB_rad'] + self.elem_size / 2, \
                            self.centred_ref_cent_coord_y[i] + self.SB_settings['SB_rad'] - self.elem_size / 2, self.elem_size)

                        elem_ref_cent_coord_xx, elem_ref_cent_coord_yy = np.meshgrid(elem_ref_cent_coord_x, elem_ref_cent_coord_y)

                        # Get averaged derivatives of the modeled Gaussian influence function
                        for j in range(self.actuator_num):                          
                            
                            self.inf_matrix_slopes[i, j] = inf_diff(elem_ref_cent_coord_xx, elem_ref_cent_coord_yy, xc, yc, j, act_diam, True)
                            self.inf_matrix_slopes[i + self.SB_settings['act_ref_cent_num'], j] = \
                                inf_diff(elem_ref_cent_coord_xx, elem_ref_cent_coord_yy, xc, yc, j, act_diam, False)
                           
                    # Take pixel size and lenslet focal length into account
                    self.inf_matrix_slopes = self.inf_matrix_slopes / self.SB_settings['pixel_size'] * config['lenslet']['lenslet_focal_length']

                    # print('Influence function is:', self.inf_matrix_slopes)

                    # Calculate singular value decomposition of influence function matrix
                    u, s, vh = np.linalg.svd(self.inf_matrix_slopes, full_matrices = False)

                    # print('u: {}, s: {}, vh: {}'.format(u, s, vh))
                    # print('The shapes of u, s, and vh are: {}, {}, and {}'.format(np.shape(u), np.shape(s), np.shape(vh)))

                    # Calculate pseudo inverse of influence function matrix to get final control matrix
                    self.control_matrix_slopes = np.linalg.pinv(self.inf_matrix_slopes)

                    svd_check_slopes = np.dot(self.control_matrix_slopes, self.inf_matrix_slopes)

                    # Get corresponding slope values generated with a unit voltage and calculated influence function matrix
                    voltages = np.zeros(self.actuator_num)
                    self.slope_x, self.slope_y = (np.zeros([2 * self.actuator_num, self.SB_settings['act_ref_cent_num']]) for i in range(2))

                    for i in range(self.actuator_num):

                        voltages_temp = voltages.copy()

                        voltages_temp[i] = config['DM']['vol_max']
                        self.slope_x[2 * i, :] = np.dot(self.inf_matrix_slopes, voltages_temp)[:self.SB_settings['act_ref_cent_num']]
                        self.slope_y[2 * i, :] = np.dot(self.inf_matrix_slopes, voltages_temp)[self.SB_settings['act_ref_cent_num']:]

                        voltages_temp[i] = config['DM']['vol_min']
                        self.slope_x[2 * i + 1, :] = np.dot(self.inf_matrix_slopes, voltages_temp)[:self.SB_settings['act_ref_cent_num']]
                        self.slope_y[2 * i + 1, :] = np.dot(self.inf_matrix_slopes, voltages_temp)[self.SB_settings['act_ref_cent_num']:]

                    print('Largest and smallest slope value for unit voltage along x axis: {}, {}'.format(np.amax(self.slope_x), np.amin(self.slope_x)))
                    print('Largest and smallest slope value for unit voltage along y axis: {}, {}'.format(np.amax(self.slope_y), np.amin(self.slope_y)))

                    self.message.emit('DM calibration process finished.')
                    # print('Control matrix is:', self.control_matrix_slopes)
                    # print('Shape of control matrix is:', np.shape(self.control_matrix_slopes))
            else:

                """
                Apply highest and lowest voltage to each actuator individually and retrieve raw slopes of each S-H spot

                Time for one calibration cycle for all actuators with image acquisition, but without centroiding: 56.686575999
                """
                # Initialise deformable mirror voltage array
                voltages = np.zeros(self.actuator_num)
                
                prev1 = time.perf_counter()

                # Create new datasets in HDF5 file to store calibration data
                get_dset(self.SB_settings, 'calibration_img', flag = 4)
                data_file = h5py.File('data_info.h5', 'a')
                data_set = data_file['calibration_img']
                
                # Poke each actuator first in to vol_max, then to vol_min
                self.message.emit('DM calibration process started...')
                for i in range(self.actuator_num):

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
                            image_max = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)

                            # Image thresholding to remove background
                            image_max = image_max - config['image']['threshold'] * np.amax(image_max)
                            image_max[image_max < 0] = 0
                            self.image.emit(image_max)

                            # Append image to list
                            dset_append(data_set, 'real_calib_img', image_max)

                            # Apply lowest voltage
                            voltages[i] = config['DM']['vol_min']

                            # Send values vector to mirror
                            self.mirror.Send(voltages)

                            # Wait for DM to settle
                            time.sleep(config['DM']['settling_time'])

                            # Acquire S-H spot image and display
                            image_min = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)

                            # Image thresholding to remove background
                            image_min = image_min - config['image']['threshold'] * np.amax(image_min)
                            image_min[image_min < 0] = 0
                            self.image.emit(image_min)

                            # Append image to list
                            dset_append(data_set, 'real_calib_img', image_min)

                            # Set actuator back to bias voltage
                            voltages[i] = config['DM']['vol_bias']
                            
                        except Exception as e:
                            print(e)
                    else:

                        self.done.emit()

                # Close HDF5 file
                data_file.close()

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
                    
                    for i in range(self.actuator_num):

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

                    svd_check_slopes = np.dot(self.control_matrix_slopes, self.inf_matrix_slopes)

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

                self.mirror_info['inf_matrix_slopes_SV'] = s
                self.mirror_info['inf_matrix_slopes'] = self.inf_matrix_slopes
                self.mirror_info['control_matrix_slopes'] = self.control_matrix_slopes
                self.mirror_info['calib_slope_x'] = self.slope_x
                self.mirror_info['calib_slope_y'] = self.slope_y
                self.mirror_info['svd_check_slopes'] = svd_check_slopes
                
                if config['dummy']:
                    self.mirror_info['act_pos_x'] = xc
                    self.mirror_info['act_pos_y'] = yc
                    self.mirror_info['act_diam'] = act_diam

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
