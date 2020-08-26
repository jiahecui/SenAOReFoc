from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import sys
import os
import argparse
import time
import click
import h5py
import numpy as np

from ximea import xiapi

import log
from config import config
from image_acquisition import acq_image
from HDF5_dset import get_mat_dset
from common import fft_spot_from_phase
from zernike_phase import zern_phase

logger = log.get_logger(__name__)

class Positioning(QObject):
    """
    Positions search blocks either through keyboard argument or import from HDF5 file
    """
    start = Signal()
    write = Signal()
    done = Signal()
    error = Signal(object)
    image = Signal(object)
    layer = Signal(object)
    message = Signal(object)
    SB_info = Signal(object)
    mirror_info = Signal(object)
    
    def __init__(self, device, settings):

        # Get search block settings
        self.SB_settings = settings

        # Get sensor instance
        self.sensor = device

        # Get sensor parameters
        self.sensor_width = self.SB_settings['sensor_width']
        self.sensor_height = self.SB_settings['sensor_height']

        # Get search block outline parameter
        self.outline_int = config['search_block']['outline_int']

        # Get information of search blocks
        self.SB_diam = self.SB_settings['SB_diam']
        self.SB_rad = self.SB_settings['SB_rad']

        # Initialise dictionary for position information
        self.pos_info = {}

        super().__init__()

    @Slot(object)
    def run(self):
        try:
            # Set process flags
            self.acquire = True
            self.inquire = True
            self.move = True
            self.load = True
            self.log = True

            # Start thread
            self.start.emit()

            """
            Acquires image and allows user to reposition search blocks by keyboard or HDF5 file
            """
            # Initialise search block layer and display initial search blocks
            self.SB_layer_2D = np.zeros([self.sensor_height, self.sensor_width])
            self.SB_layer_2D_temp = self.SB_layer_2D.copy()
            self.SB_layer_2D_temp.ravel()[self.SB_settings['act_SB_coord']] = self.outline_int

            # Acquire phase profile and retrieve S-H spot image 
            if self.acquire:

                # Acquire image
                if config['dummy']:

                    # Option 1: Load real phase profile from .mat file
                    if config['real_phase']:

                        # Retrieve real phase data and S-H spot image
                        phase = get_mat_dset(self.SB_settings, flag = 1)

                        # Get simulated S-H spots
                        self._image, self.spot_cent_x, self.spot_cent_y = fft_spot_from_phase(self.SB_settings, phase)

                    # Option 2: Leave blank if generate real zernike phase profile using DM control matrix or ideal zernike phase profile
                    else:

                        self._image = np.zeros([self.sensor_height, self.sensor_width])
                        self._image[0, 0] = self.outline_int

                        # Retrieve zernike phase map and S-H spot image
                        # zern_array = self.SB_settings['zernike_array_test']

                        # Generate ideal zernike phase profile
                        # phase = zern_phase(self.SB_settings, zern_array)

                        # Get simulated S-H spots
                        # self._image, self.spot_cent_x, self.spot_cent_y = fft_spot_from_phase(self.SB_settings, phase) 

                else:

                    self._image = acq_image(self.sensor, self.sensor_height, self.sensor_width, acq_mode = 0)               
                            
                # Image thresholding to remove background
                self._image = self._image - config['image']['threshold'] * np.amax(self._image)
                self._image[self._image < 0] = 0
                self.SB_layer_2D_temp += self._image

                self.layer.emit(self.SB_layer_2D_temp)

            else:

                self.done.emit()

            # Ask user whether DM needs calibrating, if 'y' reposition search block using keyboard, if 'n' load search block position from HDF5 file
            if self.inquire:

                self.message.emit('\nNeed to calibrate DM? [y/n]')
                c = click.getchar()

                while True:
                    if c == 'y':
                        self.load = False
                        self.message.emit('\nReposition search blocks using keyboard.')
                        break
                    elif c == 'n':
                        self.move = False
                        self.message.emit('\nLoad search block position from HDF5 file.')
                        break
                    else:
                        self.message.emit('\nInvalid input. Please try again.')

                    c = click.getchar()
            else:

                self.done.emit()

            # Roughly centre S-H spot in search block using keyboard
            if self.move:

                # Get input from keyboard to reposition search block
                self.message.emit('\nPress arrow keys to centre S-H spots in search blocks. Press Enter to finish.')
                c = click.getchar()

                # Update search block reference coordinates according to keyboard input
                while True: 
                    if c == '\xe0H' or c == '\x00H':
                        self.SB_settings['act_ref_cent_coord'] -= self.sensor_width
                        self.SB_settings['act_SB_coord'] -= self.sensor_width
                        self.SB_settings['act_ref_cent_coord_y'] -= 1
                        self.SB_settings['act_SB_offset_y'] -= 1
                    elif c == '\xe0P' or c == '\x00P':
                        self.SB_settings['act_ref_cent_coord'] += self.sensor_width
                        self.SB_settings['act_SB_coord'] += self.sensor_width
                        self.SB_settings['act_ref_cent_coord_y'] += 1
                        self.SB_settings['act_SB_offset_y'] += 1
                    elif c == '\xe0K' or c == '\x00K':
                        self.SB_settings['act_ref_cent_coord'] -= 1
                        self.SB_settings['act_SB_coord'] -= 1
                        self.SB_settings['act_ref_cent_coord_x'] -= 1
                        self.SB_settings['act_SB_offset_x'] -= 1
                    elif c == '\xe0M' or c == '\x00M':
                        self.SB_settings['act_ref_cent_coord'] += 1
                        self.SB_settings['act_SB_coord'] += 1
                        self.SB_settings['act_ref_cent_coord_x'] += 1
                        self.SB_settings['act_SB_offset_x'] += 1
                    else:
                        self.message.emit('\nSearch block position confirmed.')
                        break

                    # Display actual search blocks as they move
                    self.SB_layer_2D_temp = self.SB_layer_2D.copy()
                    self.SB_layer_2D_temp.ravel()[self.SB_settings['act_SB_coord']] = self.outline_int
                    if not config['dummy']:
                        self._image = acq_image(self.sensor, self.sensor_height, self.sensor_width, acq_mode = 0)
                        self._image = self._image - config['image']['threshold'] * np.amax(self._image)
                        self._image[self._image < 0] = 0
                    self.SB_layer_2D_temp += self._image

                    self.layer.emit(self.SB_layer_2D_temp)

                    c = click.getchar()
                    
            elif self.load:

                # Load all search block and mirror info from HDF5 file into settings
                data_file = h5py.File('data_info.h5', 'r+')
                SB_data = data_file['SB_info']
                for k in SB_data.keys():
                    self.SB_settings[k] = SB_data.get(k)[()]
                self.mirror_settings = {}
                mirror_data = data_file['mirror_info']
                for k in mirror_data.keys():
                    self.mirror_settings[k] = mirror_data.get(k)[()]   
                data_file.close()

                self.message.emit('\nSearch block position loaded.')

                # Display original search block positions from previous calibration
                self.SB_layer_2D_temp = self.SB_layer_2D.copy()
                self.SB_layer_2D_temp.ravel()[self.SB_settings['act_SB_coord']] = self.outline_int
                if not config['dummy']:
                    self._image = acq_image(self.sensor, self.sensor_height, self.sensor_width, acq_mode = 0)
                    self._image = self._image - config['image']['threshold'] * np.amax(self._image)
                    self._image[self._image < 0] = 0
                self.SB_layer_2D_temp += self._image
                self.layer.emit(self.SB_layer_2D_temp)        
            else:

                self.done.emit()

            """
            Returns position information if search block repositioned using keyboard
            """ 
            if self.log and self.move:

                self.SB_info.emit(self.SB_settings)
                self.write.emit()

            elif self.log and self.load:

                self.SB_info.emit(self.SB_settings)
                self.mirror_info.emit(self.mirror_settings)
            else:

                self.done.emit()

            # Finished positioning search blocks
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)

    @Slot()
    def stop(self):
        self.acquire = False
        self.inquire = False
        self.move = False
        self.load = False
        self.log = False


