from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import logging
import sys
import os
import imageio
import argparse
import math
import time
import click
import numpy as np

from datetime import datetime

from ximea import xiapi

import log
from config import config
from sensor import SENSOR

logger = log.get_logger(__name__)

class Centroiding(QObject):
    """
    Calculates centroids of S-H spots on camera sensor
    """
    start = Signal()
    done = Signal(object)
    error = Signal(object)
    image = Signal(object)
    layer = Signal(object)
    info = Signal(object)
    
    def __init__(self, device, settings):

        # Get search block settings
        self.SB_settings = settings

        # Get sensor instance
        self.sensor = device

        # Get camera parameters
        self.pixel_size = config['camera']['pixel_size'] * config['camera']['bin_factor']
        self.sensor_width = config['camera']['sensor_width'] // config['camera']['bin_factor']
        self.sensor_height = config['camera']['sensor_height'] // config['camera']['bin_factor']
        self.bin_factor = config['camera']['bin_factor']

        # Get search block parameter
        self.outline_int = config['search_block']['outline_int']

        super().__init__()

    def acq_image(self, acq_mode = 0):
        """
        Acquires single image or image data list according to acq_mode, 0 for single image
        """
        # Create instance of dataimage array and data list to store image data
        data = []

        # Create instance of Ximea Image to store image data and metadata
        img = xiapi.Image()

        # Open device for centroiding instance 
        self.sensor.open_device_by_SN(config['camera']['SN'])

        # Start data acquisition for each frame
        print('Starting image acquisition...')
        self.sensor.start_acquisition()
        
        if acq_mode == 0:
            # Acquire one image and display
            try:
                # Get data and pass them from camera to img
                self.sensor.get_image(img, timeout = 25)

                # Create numpy array with data from camera, dimensions are determined by imgdataformats
                dataimage = img.get_image_data_numpy()
                
                # Bin pixels to fit on S-H viewer
                dataimage = self.img_bin(dataimage, (self.sensor_width, self.sensor_height))

                # Display dataimage
                self.image.emit(dataimage)

            except xiapi.Xi_error as err:
                if err.status == 10:
                    print('Timeout error occurred.')
                else:
                    raise

        elif acq_mode == 1:
            # Acquire a sequence of images and append to data list
            for i in range(config['camera']['frame_ave_num']):
                prev1 = time.perf_counter()

                try:
                    # Get data and pass them from camera to img
                    self.sensor.get_image(img, timeout = 25)
                    prev2 = time.perf_counter()
                    print('Time for acquisition of frame {} is: {}'.format((i + 1), (prev2 - prev1)))

                    # Create numpy array with data from camera, dimensions are determined by imgdataformats
                    dataimages = img.get_image_data_numpy()

                    # Bin pixels to fit on S-H viewer
                    dataimages = self.img_bin(dataimages, (self.sensor_width, self.sensor_height))

                    # Display dataimage
                    self.image.emit(dataimages)
            
                    # Append dataimage to data list
                    data.append(dataimages)
            
                except xiapi.Xi_error as err:
                    if err.status == 10:
                        print('Timeout error occurred.')
                    else:
                        raise

                prev3 = time.perf_counter()
                print('Time for acquisition of loop {} is: {}'.format((i + 1), (prev3 - prev1)))

            print('Length of data list is:', len(data))

        # Stop data acquisition
        print('Stopping image acquisition...')
        self.sensor.stop_acquisition()

        if acq_mode == 0:
            return dataimage
        elif acq_mode == 1:
            return data
        else:
            return None

    def img_bin(self, array, new_shape):
        """
        Bins numpy arrays to form new_shape by averaging pixels
        """
        shape = (new_shape[0], array.shape[0] // new_shape[0], new_shape[1], array.shape[1] // new_shape[1])
        new_array = array.reshape(shape).mean(-1).mean(1)

        return new_array

    @Slot(object)
    def run(self):
        try:
            self.start.emit()

            """
            Acquires image and allows user to reposition search blocks
            """
            # Initialise search block layer
            self.SB_layer_2D = np.zeros([self.sensor_width, self.sensor_height], dtype='uint8')

            # Acquire S-H spot image
            self._image = self.acq_image(acq_mode = 0)

            # Get input from keyboard to reposition search block
            click.echo('Press arrow keys to centre S-H spots in search blocks.\nPress Enter to finish.', nl = False)
            c = click.getchar()

            # Update act_ref_cent_coord according to keyboard input
            while c is not '\x00d': 
                if c == '\xe0H' or c == '\x00H':
                    self.SB_settings['act_ref_cent_coord'] -= self.sensor_width
                    self.SB_settings['act_SB_coord'] -= self.sensor_width
                elif c == '\xe0P' or c == '\x00P':
                    self.SB_settings['act_ref_cent_coord'] += self.sensor_width
                    self.SB_settings['act_SB_coord'] += self.sensor_width
                elif c == '\xe0K' or c == '\x00K':
                    self.SB_settings['act_ref_cent_coord'] -= 1
                    self.SB_settings['act_SB_coord'] -= 1
                elif c == '\xe0M' or c == '\x00M':
                    self.SB_settings['act_ref_cent_coord'] += 1
                    self.SB_settings['act_SB_coord'] += 1

                # Display actual search blocks as they move
                self.SB_layer_2D_temp = self.SB_layer_2D.copy()
                self.SB_layer_2D_temp.ravel()[self.SB_settings['act_SB_coord']] = self.outline_int
                self.layer.emit(self.SB_layer_2D_temp)

                c = click.getchar()

        except Exception as e:
            raise
            self.error.emit(e)


    