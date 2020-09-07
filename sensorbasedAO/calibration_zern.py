from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import logging
import sys
import os
import argparse
import time
import numpy as np

import log
from config import config

logger = log.get_logger(__name__)

class Calibration_Zern(QObject):
    """
    Retrieves control matrix via zernikes using slopes acquired during calibration via slopes
    """
    start = Signal()
    write = Signal()
    done = Signal()
    message = Signal(object)
    error = Signal(object)
    info = Signal(object)

    def __init__(self, settings):

        # Get search block settings
        self.SB_settings = settings['SB_info']

        # Get mirror settings
        self.mirror_settings = settings['mirror_info']

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']

        # Initialise deformable mirror information parameter
        self.mirror_info = {}

        # Initialise influence function matrix
        self.inf_matrix_zern = np.zeros([config['AO']['control_coeff_num'], self.actuator_num])
        
        super().__init__()

    @Slot(object)
    def run(self):
        try:
            # Set process flags
            self.calc_inf = True
            self.log = True

            # Start thread
            self.start.emit()

            """
            Calculate individual zernike coefficients for max / min voltage of each actuator using preacquired calibration slopes
            and slope - zernike conversion matrix to create zernike influence function and control matrix
            """
            if self.calc_inf:

                # Get calibration slopes and slope - zernike conversion matrix
                self.slope_x = self.mirror_settings['calib_slope_x']
                self.slope_y = self.mirror_settings['calib_slope_y']
                self.conv_matrix =  self.mirror_settings['conv_matrix']

                # Convert slopes list to numpy array
                (self.slope_x, self.slope_y) = map(np.array, (self.slope_x, self.slope_y))

                # Concatenate x, y slopes matrix
                self.slope = np.concatenate((self.slope_x.T, self.slope_y.T), axis = 0)

                # Fill influence function matrix by multiplying each column in slopes matrix with the conversion matrix
                for i in range(self.actuator_num):

                    self.inf_matrix_zern[:, i] = \
                        np.dot(self.conv_matrix, (self.slope[:, 2 * i] - self.slope[:, 2 * i + 1]))[:config['AO']['control_coeff_num']] \
                            / (config['DM']['vol_max'] - config['DM']['vol_min'])
            
                # Get singular value decomposition of influence function matrix
                u, s, vh = np.linalg.svd(self.inf_matrix_zern, full_matrices = False)

                # print('u: {}, s: {}, vh: {}'.format(u, s, vh))
                # print('The shapes of u, s, and vh are: {}, {}, and {}'.format(np.shape(u), np.shape(s), np.shape(vh)))
                
                # Calculate pseudo inverse of influence function matrix to get final control matrix
                self.control_matrix_zern = np.linalg.pinv(self.inf_matrix_zern, rcond = 0.06)

                # u1, s1, vh1 = np.linalg.svd(self.control_matrix_zern, full_matrices = False)
                # print('u1: {}, s1: {}, vh1: {}'.format(u1, s1, vh1))

                svd_check_zern = np.dot(self.inf_matrix_zern, self.control_matrix_zern)

                print('Largest Zernike amplitude for unit actuator voltage: {}\n'.format(np.amax(self.inf_matrix_zern)))

                self.message.emit('\nZernike control matrix retrieved.')
                # print('Control matrix is:', self.control_matrix_zern)
                # print('Shape of control matrix is:', np.shape(self.control_matrix_zern))
            else:

                self.done.emit()

            """
            Returns zernike calibration information into self.mirror_info
            """ 
            if self.log:

                self.mirror_info['inf_matrix_zern_SV'] = s
                self.mirror_info['inf_matrix_zern'] = self.inf_matrix_zern
                self.mirror_info['control_matrix_zern'] = self.control_matrix_zern
                self.mirror_info['svd_check_zern'] = svd_check_zern

                self.info.emit(self.mirror_info)
                self.write.emit()
            else:

                self.done.emit()

            # Finished retrieving zernike influence function and control matrix
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)

    @Slot(object)
    def stop(self):
        self.calc_inf = False
        self.log = False
