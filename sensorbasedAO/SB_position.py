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
from sensor import SENSOR
from image_acquisition import acq_image
from spot_sim import SpotSim

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
    info = Signal(object)
    
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
            self.SB_layer_2D = np.zeros([self.sensor_width, self.sensor_height], dtype='uint8')
            self.SB_layer_2D_temp = self.SB_layer_2D.copy()
            self.SB_layer_2D_temp.ravel()[self.SB_settings['act_SB_coord']] = self.outline_int
            self.layer.emit(self.SB_layer_2D_temp)

            # Acquire S-H spot image or use simulated Gaussian profile S-H spot image
            if self.acquire:
                # Acquire image
                # self._image = acq_image(self.sensor, self.sensor_width, self.sensor_height, acq_mode = 0)
                spot_img = SpotSim(self.SB_settings)
                self._image, self.spot_cent_x, self.spot_cent_y = spot_img.SH_spot_sim(centred = 0)

                # Image thresholding to remove background
                self._image = self._image - config['image']['threshold'] * np.amax(self._image)
                self._image[self._image < 0] = 0
                self.image.emit(self._image)
            else:
                self.done.emit()

            # Ask user whether DM needs calibrating, if 'y' reposition search block using keyboard, if 'n' load search block position from HDF5 file
            if self.inquire:

                self.message.emit('Need to calibrate DM? [y/n]')
                c = click.getchar()

                while True:
                    if c == 'y':
                        self.load = False
                        self.message.emit('Reposition search blocks using keyboard.')
                        break
                    elif c == 'n':
                        self.move = False
                        self.message.emit('Load search block position from HDF5 file.')
                        break
                    else:
                        self.message.emit('Invalid input. Try again.')

                    c = click.getchar()
            else:

                self.done.emit()

            # Roughly centre S-H spot in search block using keyboard
            if self.move:

                # Get input from keyboard to reposition search block
                self.message.emit('Press arrow keys to centre S-H spots in search blocks. Press Enter to finish.')
                c = click.getchar()

                # Update search block reference coordinates according to keyboard input
                while True: 
                    if c == '\xe0H' or c == '\x00H':
                        self.SB_settings['act_ref_cent_coord'] -= self.sensor_width
                        self.SB_settings['act_SB_coord'] -= self.sensor_width
                        self.SB_settings['act_ref_cent_coord_y'] -= 1
                    elif c == '\xe0P' or c == '\x00P':
                        self.SB_settings['act_ref_cent_coord'] += self.sensor_width
                        self.SB_settings['act_SB_coord'] += self.sensor_width
                        self.SB_settings['act_ref_cent_coord_y'] += 1
                    elif c == '\xe0K' or c == '\x00K':
                        self.SB_settings['act_ref_cent_coord'] -= 1
                        self.SB_settings['act_SB_coord'] -= 1
                        self.SB_settings['act_ref_cent_coord_x'] -= 1
                    elif c == '\xe0M' or c == '\x00M':
                        self.SB_settings['act_ref_cent_coord'] += 1
                        self.SB_settings['act_SB_coord'] += 1
                        self.SB_settings['act_ref_cent_coord_x'] += 1
                    else:
                        self.message.emit('Search block position confirmed.')
                        break

                    # Display actual search blocks as they move
                    self.SB_layer_2D_temp = self.SB_layer_2D.copy()
                    self.SB_layer_2D_temp.ravel()[self.SB_settings['act_SB_coord']] = self.outline_int
                    self.layer.emit(self.SB_layer_2D_temp)

                    c = click.getchar()
            elif self.load:

                # Load search block position from HDF5 file
                data_file = h5py.File('data_info.h5', 'r+')
                data = data_file['SB_info']
                self.SB_settings['act_ref_cent_coord'] = data.get('act_ref_cent_coord')[()]
                self.SB_settings['act_ref_cent_coord_x'] = data.get('act_ref_cent_coord_x')[()]
                self.SB_settings['act_ref_cent_coord_y'] = data.get('act_ref_cent_coord_y')[()]
                self.SB_settings['act_SB_coord'] = data.get('act_SB_coord')[()]
                data_file.close()

                self.message.emit('Search block position loaded.') 

                # Display original search block positions from previous calibration
                self.SB_layer_2D_temp = self.SB_layer_2D.copy()
                self.SB_layer_2D_temp.ravel()[self.SB_settings['act_SB_coord']] = self.outline_int
                self.layer.emit(self.SB_layer_2D_temp)
            else:

                self.done.emit()

            """
            Returns position information if search block repositioned using keyboard
            """ 
            if self.log and self.move:

                self.info.emit(self.SB_settings)
                self.write.emit()
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


