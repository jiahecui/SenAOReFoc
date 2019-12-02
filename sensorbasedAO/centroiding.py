from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import sys
import os
import argparse
import time
import click
import numpy as np

from ximea import xiapi

import log
from config import config
from sensor import SENSOR
from image_acquisition import acq_image
from spot_sim import SpotSim

logger = log.get_logger(__name__)

class Centroiding(QObject):
    """
    Calculates centroids of S-H spots on camera sensor
    """
    start = Signal()
    done = Signal()
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

        # Initialise dictionary for centroid information
        self.cent_info = {}

        super().__init__()

    @Slot(object)
    def run(self):
        try:
            """
            Calculate actual S-H spot centroid coordinates
            """
            # Get actual SB_layer_2D array
            self.SB_layer_2D = self.SB_layer_2D_temp.copy()

            # Get actual search block reference centroid coords
            self.act_ref_cent_coord = self.SB_settings['act_ref_cent_coord']
            self.act_ref_cent_coord_x = self.SB_settings['act_ref_cent_coord_x']
            self.act_ref_cent_coord_y = self.SB_settings['act_ref_cent_coord_y']
            self.act_SB_coord = self.SB_settings['act_SB_coord']

            # print('Act_ref_cent_coord_x:', self.act_ref_cent_coord_x)
            # print('Act_ref_cent_coord_y:', self.act_ref_cent_coord_y)
            # print(self.SB_layer_2D.ravel()[int(self.act_ref_cent_coord[1] - 50) : int(self.act_ref_cent_coord[1] + 50)])

            # Initialise actual S-H spot centroid coords array
            self.act_cent_coord, self.act_cent_coord_x, self.act_cent_coord_y = (np.zeros(len(self.act_ref_cent_coord)) for i in range(3))

            # Calculate actual S-H spot centroids for each search block using a dynamic range
            if self.calculate:
                
                prev1 = time.perf_counter()

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
                            """
                            Two methods for setting the dynamic range

                            Notes:
                                1) Without thresholding and both doing 5 cycles, Method 2 is 2 - 3 times more effective for uniform noise below 5 
                                    (low noise level) and slightly (1 - 2 times) more effective for uniform noise above 7.
                                2) Without thresholding and both doing 5 cycles, Method 2 is 2 times more effective for Gaussian, Method 1 is slightly 
                                    more effective for speckle, both are equally effective for Poisson.
                                3) Method 2 is much more stable than Method 1 (error level using Method 1 sometimes double with the same parameters).
                                4) The size of each S-H spot affects centroiding accuracy, the smaller the spot (smaller sigma), the less accurate the
                                    centroiding is with all other parameters the same.
                                5) Using Method 2, without thresholding and doing 5 cycles, the average positioning error is around 0.5 for a 
                                    uniform noise level of 7 and sigma of 4.                                    
                                6) Using Method 2, without dynamic range and using thresholding value of 0.1, the average positioning error is around 
                                    0.002 for a uniform noise level below 28 and sigma of 2 - 4, around 0.3 for a uniform noise level of 29 and sigma
                                    of 4, and around 0.6 for a uniform noise level of 30 and sigma of 4. However, the average positioning error increases 
                                    rapidly to 0.8 for a uniform noise level of 29 and sigma of 2, and 1.5 for a uniform noise level of 30 and sigma of 2.
                                7) Using Method 2, without dynamic range and using thresholding value of 0.1, the positions (randomness) of each S-H spot 
                                    WITHIN the search block does not affect centroiding accuracy, but being on the lines has substantial affect. Using 2 
                                    cycles of dynamic range alleviates this affect greatly (to the same accuracy as when all spots are WITHIN the search
                                    blocks). In this case, also using thresholding value of 0.1, the average positioning error is around 0.17 for a uniform
                                    noise level of 30 and sigma of 4 when all spots are WITHIN the search blocks.                                
                                8) With 180 spots, the time for one centroiding process is around 1.3 s for 1 cycle, 2.5 s for 2 cycles, and 5.5 s for 
                                    5 cycles of dynamic range.                                 
                            """
                            # Method 1: Judge the position of S-H spot centroid relative to centre of search block and decrease dynamic range
                            # if self.act_cent_coord_x[i] > self.act_ref_cent_coord_x[i]:
                            #     SB_pix_coord_x = SB_pix_coord_x[1:]
                            # elif self.act_cent_coord_x[i] < self.act_ref_cent_coord_x[i]:
                            #     SB_pix_coord_x = SB_pix_coord_x[:-1]
                            # else:
                            #     SB_pix_coord_x = SB_pix_coord_x[1:-1]

                            # if self.act_cent_coord_y[i] > self.act_ref_cent_coord_y[i]:
                            #     SB_pix_coord_y = SB_pix_coord_y[1:]
                            # elif self.act_cent_coord_y[i] < self.act_ref_cent_coord_y[i]:
                            #     SB_pix_coord_y = SB_pix_coord_y[:-1]
                            # else:
                            #     SB_pix_coord_y = SB_pix_coord_y[1:-1]

                            # Method 2: Centre new search area on centroid calculated during previous cycle while shrinking search area at the same time
                            SB_pix_coord_x = np.arange(self.act_cent_coord_x[i] - self.SB_rad + 1 + n, \
                                self.act_cent_coord_x[i] + self.SB_rad - 1 - n)
                            SB_pix_coord_y = np.arange(self.act_cent_coord_y[i] - self.SB_rad + 1 + n, \
                                self.act_cent_coord_y[i] + self.SB_rad - 1 - n)

                        # if i == 0:
                        #     print('SB_pixel_coord_x_{}_{}: {}'.format(i, n, SB_pix_coord_x))
                        #     print('SB_pixel_coord_y_{}_{}: {}'.format(i, n, SB_pix_coord_y))
                        #     print('Length of pixel coord along x axis for cycle {}: {}'.format(n, len(SB_pix_coord_x)))
                        #     print('Length of pixel coord along y axis for cycle {}: {}'.format(n, len(SB_pix_coord_y)))

                        # Calculate actual S-H spot centroids by doing weighted sum
                        for j in range(len(SB_pix_coord_y)):
                            for k in range(len(SB_pix_coord_x)):
                                sum_x += self._image[int(SB_pix_coord_y[j]), int(SB_pix_coord_x[k])] * int(SB_pix_coord_x[k])
                                sum_y += self._image[int(SB_pix_coord_y[j]), int(SB_pix_coord_x[k])] * int(SB_pix_coord_y[j])
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

                prev2 = time.perf_counter()
                print('Time for one centroiding process is:', (prev2 - prev1))

                # print('Act_cent_coord_x:', self.act_cent_coord_x)
                # print('Act_cent_coord_y:', self.act_cent_coord_y)
                # print('Act_cent_coord:', self.act_cent_coord)
                # print('Error along x axis:', error_x)
                # print('Error along y axis:', error_y)
                print('Average position error:', error_tot)
                # print('Slope along x axis:', self.slope_x)
                # print('Slope along y axis:', self.slope_y)

                # Draw actual S-H spot centroids on image layer
                self._image.ravel()[self.act_cent_coord.astype(int)] = 0
                self.image.emit(self._image)
            else:

                self.done.emit()

            """
            Returns centroid information into self.cent_info
            """ 
            if self.log:

                self.cent_info['act_ref_cent_coord_x'] = self.act_ref_cent_coord_x
                self.cent_info['act_ref_cent_coord_y'] = self.act_ref_cent_coord_y
                self.cent_info['act_ref_cent_coord'] = self.act_ref_cent_coord
                self.cent_info['act_SB_coord'] = self.act_SB_coord
                self.cent_info['act_cent_coord_x'] = self.act_cent_coord_x
                self.cent_info['act_cent_coord_y'] = self.act_cent_coord_y
                self.cent_info['act_cent_coord'] = self.act_cent_coord
                self.cent_info['slope_x'] = self.slope_x
                self.cent_info['slope_y'] = self.slope_y

                self.info.emit(self.cent_info)
            else:

                self.done.emit()
       
            # Finished calculating centroids of S-H spots
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)

    @Slot()
    def stop(self):
        self.acquire = False
        self.move = False
        self.calculate = False
        self.log = False


    