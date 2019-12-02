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
from centroid_acquisition import acq_centroid
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

        # Initialise data lists to pass into centroid_acquisition.py
        self.data, self.cent_x, self.cent_y = ([] for i in range(3))

        # Initialise actual S-H spot centroid coords array
        self.act_cent_coord, self.act_cent_coord_x, self.act_cent_coord_y = (np.zeros(self.SB_settings['act_ref_cent_num']) for i in range(3))

        # Initialise dictionary for centroid information
        self.cent_info = {}

        super().__init__()

    @Slot(object)
    def run(self):
        try:
            # Set process flags
            self.calc_cent = True
            self.log = True

            # Start thread
            self.start.emit()

            """
            Calculate actual S-H spot centroid coordinates
            """
            # Initialise search block layer and display search blocks
            self.SB_layer_2D = np.zeros([self.sensor_width, self.sensor_height], dtype = 'uint8')
            self.SB_layer_2D_temp = self.SB_layer_2D.copy()
            self.SB_layer_2D_temp.ravel()[self.SB_settings['act_SB_coord']] = self.outline_int
            self.layer.emit(self.SB_layer_2D_temp)

            # Acquire image using sensor or simulate Gaussian profile S-H spots
            # self._image = acq_image(self.sensor, self.sensor_width, self.sensor_height, acq_mode = 0)
            spot_img = SpotSim(self.SB_settings)
            self._image, self.spot_cent_x, self.spot_cent_y = spot_img.SH_spot_sim(centred = 1)

            # Image thresholding to remove background
            self._image = self._image - config['image']['threshold'] * np.amax(self._image)
            self._image[self._image < 0] = 0
            self.image.emit(self._image)
            
            # Append data to data list
            self.data.append(self._image)
            self.cent_x.append(self.spot_cent_x)
            self.cent_y.append(self.spot_cent_y)

            # Calculate centroids for S-H spots
            if self.calc_cent:
                
                # Acquire centroid information
                self.act_cent_coord, self.act_cent_coord_x, self.act_cent_coord_y, self.slope_x, self.slope_y = \
                    acq_centroid(self.SB_settings, self.cent_x, self.cent_y, self.data)
                self.act_cent_coord, self.act_cent_coord_x, self.act_cent_coord_y, self.slope_x, self.slope_y = \
                    map(np.asarray, [self.act_cent_coord, self.act_cent_coord_x, self.act_cent_coord_y, self.slope_x, self.slope_y])

                # Draw actual S-H spot centroids on image layer
                self._image.ravel()[self.act_cent_coord.astype(int)] = 0
                self.image.emit(self._image)
            else:

                self.done.emit()

            """
            Returns centroid information into self.cent_info
            """ 
            if self.log:

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
        self.calc_cent = False
        self.log = False


    