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
from zernike import zern, zern_diff

logger = log.get_logger(__name__)

class Conversion(QObject):
    """
    Generates slope - zernike conversion matrix and zernike matrix for a given pupil shape
    """
    start = Signal()
    write = Signal()
    done = Signal()
    message = Signal(object)
    error = Signal(object)
    info = Signal(object)

    def __init__(self, settings):

        # Get search block settings
        self.SB_settings = settings

        # Initialise sensor parameters
        self.sensor_width = self.SB_settings['sensor_width']

        # Initialise search block parameters
        self.SB_rad = self.SB_settings['SB_rad']
        self.SB_across_width = self.SB_settings['SB_across_width']
        self.act_ref_cent_coord_x = self.SB_settings['act_ref_cent_coord_x']
        self.act_ref_cent_coord_y = self.SB_settings['act_ref_cent_coord_y']

        # Initialise conversion matrix information parameter
        self.conv_info = {}

        # Initialise zernike matrix and zernike derivative matrix
        self.zern_matrix = np.zeros([self.SB_settings['act_ref_cent_num'], config['AO']['recon_coeff_num']])
        self.diff_matrix = np.zeros([2 * self.SB_settings['act_ref_cent_num'], config['AO']['recon_coeff_num']]) 
        
        super().__init__()

    @Slot(object)
    def run(self):
        try:
            # Set process flags
            self.norm_coords = True
            self.calculate = True
            self.log = True

            # Start thread
            self.start.emit()

            """
            Get normalised reference coordinates in unit circle
            """
            if self.norm_coords:

                # Get rescale factor between normalised coordinates and actual coordinates
                self.rescale = 2 / (config['search_block']['pupil_diam'] * 1e3 / self.SB_settings['pixel_size'])

                # Get normalised search block radius
                self.norm_rad = self.SB_rad * self.rescale

                # Get size of each individual element in unit circle
                self.elem_size = self.SB_settings['SB_diam'] / config['search_block']['div_elem'] * self.rescale

                # print('Rescale: {}, norm_rad: {}, elem_size: {}'.format(self.rescale, self.norm_rad, self.elem_size))
                
                # Get reference centroid coordinates for unit circle
                if (self.SB_across_width % 2 == 0 and self.sensor_width % 2 == 0) or \
                    (self.SB_across_width % 2 == 1 and self.sensor_width % 2 == 1):

                    self.norm_ref_cent_coord_x = (self.act_ref_cent_coord_x - self.sensor_width // 2) * self.rescale
                    self.norm_ref_cent_coord_y = (self.act_ref_cent_coord_y - self.sensor_width // 2) * self.rescale

                else:

                    self.norm_ref_cent_coord_x = (self.act_ref_cent_coord_x - (self.sensor_width // 2 - self.SB_rad)) * self.rescale
                    self.norm_ref_cent_coord_y = (self.act_ref_cent_coord_y - (self.sensor_width // 2 - self.SB_rad)) * self.rescale

                # Take account of odd number of relays and mirrors between DM and lenslet
                if config['relay']['mirror_odd']:
                    self.norm_ref_cent_coord_x = -self.norm_ref_cent_coord_x

                if config['relay']['relay_odd']:
                    self.norm_ref_cent_coord_x = -self.norm_ref_cent_coord_x
                    self.norm_ref_cent_coord_y = -self.norm_ref_cent_coord_y

                # print('Norm_ref_cent_coord_x:', self.norm_ref_cent_coord_x)
                # print('Norm_ref_cent_coord_y:', self.norm_ref_cent_coord_y)
            else:

                self.done.emit()

            """
            Get normalised coordinates for each individual element in search block to calculate zernike matrix and conversion matrix
            """
            if self.calculate:

                self.message.emit('Retrieving zernike matrix and slope - zernike conversion matrix...')
                for i in range(self.SB_settings['act_ref_cent_num']):

                    # Get reference centroid coords of each element
                    elem_ref_cent_coord_x = np.arange(self.norm_ref_cent_coord_x[i] - self.norm_rad + self.elem_size / 2, \
                        self.norm_ref_cent_coord_x[i] + self.norm_rad - self.elem_size / 2, self.elem_size)
                    elem_ref_cent_coord_y = np.arange(self.norm_ref_cent_coord_y[i] - self.norm_rad + self.elem_size / 2, \
                        self.norm_ref_cent_coord_y[i] + self.norm_rad - self.elem_size / 2, self.elem_size)

                    # print('Elem_ref_cent_coord_x:', elem_ref_cent_coord_x)
                    # print('Elem_ref_cent_coord_y:', elem_ref_cent_coord_y)

                    # Get averaged x and y values and derivatives of the jth Zernike polynomial to fill zernike matrix and 
                    # zernike derivative matrix respectively 
                    for j in range(config['AO']['recon_coeff_num']):

                        self.zern_matrix[i, j] = zern(elem_ref_cent_coord_x, elem_ref_cent_coord_y, j + 1)

                        self.diff_matrix[i, j] = zern_diff(elem_ref_cent_coord_x, elem_ref_cent_coord_y, j + 1, True)
                        self.diff_matrix[i + self.SB_settings['act_ref_cent_num'], j] = zern_diff(elem_ref_cent_coord_x, elem_ref_cent_coord_y, j + 1, False)

                # Get singular value decomposition of zernike derivative matrix
                u, s, vh = np.linalg.svd(self.diff_matrix, full_matrices = False)

                # print('u: {}, s: {}, vh: {}'.format(u, s, vh))
                # print('The shapes of u, s, and vh are: {}, {}, and {}'.format(np.shape(u), np.shape(s), np.shape(vh)))
                
                # Calculate pseudo inverse of zernike derivative matrix to get conversion matrix
                self.conv_matrix = np.linalg.pinv(self.diff_matrix)

                svd_check_conv = np.dot(self.conv_matrix, self.diff_matrix)

                # Modify unit of conversion matrix from None to um / pixel in order to convert from pixels -> um during slope - zernike conversion
                self.conv_matrix = self.conv_matrix * self.SB_settings['pixel_size']

                self.message.emit('Zernike matrix and slope - zernike conversion matrix retrieved.')
                # print('Conversion matrix is:', self.conv_matrix)
                # print('Shape of conversion matrix is:', np.shape(self.conv_matrix))
            else:

                self.done.emit()

            """
            Returns zernike matrix and slope - zernike conversion matrix information into self.conv_info
            """ 
            if self.log:

                self.conv_info['norm_ref_cent_coord_x'] = self.norm_ref_cent_coord_x
                self.conv_info['norm_ref_cent_coord_y'] = self.norm_ref_cent_coord_y
                self.conv_info['conv_matrix_SV'] = s
                self.conv_info['zern_matrix'] = self.zern_matrix
                self.conv_info['diff_matrix'] = self.diff_matrix
                self.conv_info['conv_matrix'] = self.conv_matrix
                self.conv_info['svd_check_conv'] = svd_check_conv

                self.info.emit(self.conv_info)
                self.write.emit()
            else:

                self.done.emit()

            # Finished retrieving slope - zernike conversion matrix
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)

    @Slot(object)
    def stop(self):
        self.norm_coords = False
        self.calculate = False
        self.log = False

                



