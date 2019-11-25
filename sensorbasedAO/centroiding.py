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
from spot_sim import SpotSim

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

        # Get sensor parameters
        self.sensor_width = self.SB_settings['sensor_width']
        self.sensor_height = self.SB_settings['sensor_height']

        # Get search block outline parameter
        self.outline_int = config['search_block']['outline_int']

        # Get information of search blocks
        self.SB_diam = self.SB_settings['SB_diam']
        self.SB_rad = self.SB_settings['SB_rad']

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
            # Initialise search block layer and display initial search blocks
            self.SB_layer_2D = np.zeros([self.sensor_width, self.sensor_height], dtype='uint8')
            self.SB_layer_2D_temp = self.SB_layer_2D.copy()
            self.SB_layer_2D_temp.ravel()[self.SB_settings['act_SB_coord']] = self.outline_int
            self.layer.emit(self.SB_layer_2D_temp)

            # Acquire S-H spot image or use simulated Gaussian profile S-H spot image 
            # self._image = self.acq_image(acq_mode = 0)
            spot_img = SpotSim(self.SB_settings)
            self._image, self.spot_cent_x, self.spot_cent_y = spot_img.SH_spot_sim()

            # Image thresholding to remove background
            self._image = self._image - config['image']['threshold'] * np.amax(self._image)
            self._image[self._image < 0] = 0
            self.image.emit(self._image)

            # Get input from keyboard to reposition search block
            click.echo('Press arrow keys to centre S-H spots in search blocks.\nPress Enter to finish.', nl = False)
            c = click.getchar()

            # Update act_ref_cent_coord according to keyboard input
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
                    break

                # Display actual search blocks as they move
                self.SB_layer_2D_temp = self.SB_layer_2D.copy()
                self.SB_layer_2D_temp.ravel()[self.SB_settings['act_SB_coord']] = self.outline_int
                self.layer.emit(self.SB_layer_2D_temp)

                c = click.getchar()

            """
            Calculate actual S-H spot centroid coordinates
            """
            # Get actual SB_layer_2D array
            self.SB_layer_2D = self.SB_layer_2D_temp.copy()

            # Get actual search block reference centroid coords
            self.act_ref_cent_coord = self.SB_settings['act_ref_cent_coord']
            self.act_ref_cent_coord_x = self.SB_settings['act_ref_cent_coord_x']
            self.act_ref_cent_coord_y = self.SB_settings['act_ref_cent_coord_y']

            # print('Act_ref_cent_coord_x:', self.act_ref_cent_coord_x)
            # print('Act_ref_cent_coord_y:', self.act_ref_cent_coord_y)
            # print(self.SB_layer_2D.ravel()[int(self.act_ref_cent_coord[1] - 50) : int(self.act_ref_cent_coord[1] + 50)])

            # Initialise actual S-H spot centroid coords array
            self.act_cent_coord, self.act_cent_coord_x, self.act_cent_coord_y = (np.zeros(len(self.act_ref_cent_coord)) for i in range(3))

            # Calculate actual S-H spot centroids for each search block using a dynamic range 
            for i in range(len(self.act_ref_cent_coord)):

                for n in range(config['image']['dynamic_num']):

                    # Initialise temporary summing parameters
                    sum_x = 0
                    sum_y = 0
                    sum_pix = 0

                    if n == 0:
                        # Get 2D coords of pixels in each search block that need to be summed
                        if self.SB_settings['odd_pix']:
                            SB_pix_coord_x = np.arange(self.act_ref_cent_coord_x[i] - self.SB_rad + 1, \
                                self.act_ref_cent_coord_x[i] + self.SB_rad - 1)
                            SB_pix_coord_y = np.arange(self.act_ref_cent_coord_y[i] - self.SB_rad + 1, \
                                self.act_ref_cent_coord_y[i] + self.SB_rad - 1)
                        else:
                            SB_pix_coord_x = np.arange(self.act_ref_cent_coord_x[i] - self.SB_rad + 2, \
                                self.act_ref_cent_coord_x[i] + self.SB_rad - 1)
                            SB_pix_coord_y = np.arange(self.act_ref_cent_coord_y[i] - self.SB_rad + 2, \
                                self.act_ref_cent_coord_y[i] + self.SB_rad - 1)
                    else:
                        # Judge the position of S-H spot centroid relative to centre of search block and decrease dynamic range
                        if self.act_cent_coord_x[i] > self.act_ref_cent_coord_x[i]:
                            SB_pix_coord_x = SB_pix_coord_x[1:]
                        elif self.act_cent_coord_x[i] < self.act_ref_cent_coord_x[i]:
                            SB_pix_coord_x = SB_pix_coord_x[:-1]
                        else:
                            SB_pix_coord_x = SB_pix_coord_x[1:-1]

                        if self.act_cent_coord_y[i] > self.act_ref_cent_coord_y[i]:
                            SB_pix_coord_y = SB_pix_coord_y[1:]
                        elif self.act_cent_coord_y[i] < self.act_ref_cent_coord_y[i]:
                            SB_pix_coord_y = SB_pix_coord_y[:-1]
                        else:
                            SB_pix_coord_y = SB_pix_coord_y[1:-1]

                    # if i == 0:
                    #     print('SB_pixel_coord_x_{}_{}: {}'.format(i, n, SB_pix_coord_x))
                    #     print('SB_pixel_coord_y_{}_{}: {}'.format(i, n, SB_pix_coord_y))
                    #     print('Length of pixel coord along x axis for cycle {}: {}'.format(n, len(SB_pix_coord_x)))
                    #     print('Length of pixel coord along y axis for cycle {}: {}'.format(n, len(SB_pix_coord_y)))

                    # Calculate actual S-H spot centroids by doing weighted sum
                    for j in range(len(SB_pix_coord_y)):
                        for k in range(len(SB_pix_coord_x)):
                            sum_x += self._image[int(SB_pix_coord_y[j]), int(SB_pix_coord_x[k])] * SB_pix_coord_x[k]
                            sum_y += self._image[int(SB_pix_coord_y[j]), int(SB_pix_coord_x[k])] * SB_pix_coord_y[j]
                            sum_pix += self._image[int(SB_pix_coord_y[j]), int(SB_pix_coord_x[k])]

                    self.act_cent_coord_x[i] = sum_x / sum_pix
                    self.act_cent_coord_y[i] = sum_y / sum_pix
                    self.act_cent_coord[i] = int(self.act_cent_coord_y[i]) * self.sensor_width + int(self.act_cent_coord_x[i])                          

            # Calculate average centroid error 
            error_temp = 0
            error_x = self.act_cent_coord_x - self.spot_cent_x
            error_y = self.act_cent_coord_y - self.spot_cent_y
            for i in range(len(error_x)):
                error_temp += np.sqrt(error_x[i] ** 2 + error_y[i] ** 2)
            error_tot = error_temp / len(error_x)

            # Calculate raw slopes in each dimension
            self.slope_x = self.act_cent_coord_x - self.act_ref_cent_coord_x
            self.slope_y = self.act_cent_coord_y - self.act_ref_cent_coord_y

            # print('Act_cent_coord_x:', self.act_cent_coord_x)
            # print('Act_cent_coord_y:', self.act_cent_coord_y)
            # print('Act_cent_coord:', self.act_cent_coord)
            # print('Error along x axis:', error_x)
            # print('Error along y axis:', error_y)
            # print('Average position error:', error_tot)
            # print('Slope along x axis:', self.slope_x)
            # print('Slope along y axis:', self.slope_y)

            # Draw actual S-H spot centroids on image layer
            self.SB_layer_2D.ravel()[self.act_cent_coord.astype(int)] = self.outline_int
            self.layer.emit(self.SB_layer_2D)

            """
            Returns centroid information into self.SB_settings
            """
            self.SB_settings['act_ref_cent_coord_x'] = self.act_ref_cent_coord_x
            self.SB_settings['act_ref_cent_coord_y'] = self.act_ref_cent_coord_y
            self.SB_settings['act_ref_cent_coord'] = self.act_ref_cent_coord
            self.SB_settings['act_SB_coord'] = self.act_SB_coord
            self.SB_settings['act_cent_coord_x'] = self.act_cent_coord_x
            self.SB_settings['act_cent_coord_y'] = self.act_cent_coord_y
            self.SB_settings['act_cent_coord'] = self.act_cent_coord
            self.SB_settings['slope_x'] = self.slope_x
            self.SB_settings['slope_y'] = self.slope_y

            self.info.emit(self.SB_settings)

            # Finished calculating centroids of S-H spots
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)


    