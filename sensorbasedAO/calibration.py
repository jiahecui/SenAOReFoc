from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import logging
import sys
import os
import argparse
import time
import PIL.Image
import numpy as np

import log
from config import config
from image_acquisition import acq_image

logger = log.get_logger(__name__)

class Calibration(QObject):
    """
    Calibrates deformable mirror and retrieves influence function
    """
    start = Signal()
    done = Signal()
    error = Signal(object)
    image = Signal(object)
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

    def calib_centroid(self):

        return slope_x, slope_y

    @Slot(object)
    def run(self):
        try:
            self.start.emit()

            """
            Apply highest and lowest voltage to each actuator individually and retrieve raw slopes of each S-H spot
            """
            for i in range(config['DM']['actuator_num']):

                # Apply highest voltage
                voltages[i] = config['DM']['vol_max']

                # Send values vector to mirror
                self.mirror.Send(voltages)

                # Wait for DM to settle
                time.sleep(config['DM']['settling_time'])

                # Acquire S-H spot image and display
                image_max = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)
                self.image.emit(image_max)

                # Calculate S-H spot centroid coordinates to get slopes
                slope_x_max, slope_y_max = calib_centroid()

                # Apply lowest voltage
                voltages[i] = config['DM']['vol_min']

                # Send values vector to mirror
                self.mirror.Send(voltages)

                # Wait for DM to settle
                time.sleep(config['DM']['settling_time'])

                # Acquire S-H spot image and display
                image_min = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)
                self.image.emit(image_min)

                # Calculate S-H spot centroid coordinates to get slopes
                slope_x_min, slope_y_min = calib_centroid()

                # Set actuator back to bias voltage
                voltages[i] = config['DM']['vol_bias']

                # Send values vector to mirror
                self.mirror.Send(voltages)

                # Fill influence function matrix with acquired slopes
                self.inf_matrix_slopes[:self.SB_settings['act_ref_cent_num'] - 1, i] = \
                    (slope_x_max - slope_x_min) / (config['DM']['vol_max'] - config['DM']['vol_min'])
                self.inf_matrix_slopes[self.SB_settings['act_ref_cent_num']:, i] = \
                    (slope_y_max - slope_y_min) / (config['DM']['vol_max'] - config['DM']['vol_min'])

            """
            Returns deformable mirror calibration information into self.mirror_info
            """ 
            self.mirror_info['inf_matrix_slopes'] = self.inf_matrix_slopes

            self.info.emit(self.mirror_info)

            # Finished calibrating deformable mirror and retrieving influence functions
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)
