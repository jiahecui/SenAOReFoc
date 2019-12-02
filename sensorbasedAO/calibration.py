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
from centroid_acquisition import acq_centroid
from spot_sim import SpotSim

logger = log.get_logger(__name__)

class Calibration(QObject):
    """
    Calibrates deformable mirror and retrieves influence function
    """
    start = Signal()
    done = Signal()
    error = Signal(object)
    image = Signal(object)
    cent = Signal(object)
    info = Signal(object)

    def __init__(self, sensor, mirror, settings):

        # Get search block settings
        self.SB_settings = settings

        # Get sensor instance
        self.sensor = sensor

        # Get mirror instance
        self.mirror = mirror

        # Initialise data lists to pass into centroid_acquisition.py
        self.data, self.cent_x, self.cent_y = ([] for i in range(3))

        # Initialise deformable mirror information parameter
        self.mirror_info = {}

        # Initialise influence function matrix
        self.inf_matrix_slopes = np.zeros([2 * self.SB_settings['act_ref_cent_num'], config['DM']['actuator_num']])
        
        super().__init__()

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

            """
            Apply highest and lowest voltage to each actuator individually and retrieve raw slopes of each S-H spot

            Time for one calibration cycle for all actuators with image acquisition, but without centroiding: 56.686575999
            """
            # Initialise deformable mirror voltage array
            voltages = np.zeros(config['DM']['actuator_num'])
            
            prev1 = time.perf_counter()

            print('Actuator calibration process started...')
            for i in range(config['DM']['actuator_num']):

                if self.calibrate:

                    # Apply highest voltage
                    voltages[i] = config['DM']['vol_max']
                
                    # Send values vector to mirror
                    self.mirror.Send(voltages)
                    
                    # Wait for DM to settle
                    time.sleep(config['DM']['settling_time'])
                    
                    # Acquire S-H spot image and display
                    # image_max = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)
                    spot_img = SpotSim(self.SB_settings)
                    image_max, self.spot_cent_x, self.spot_cent_y = spot_img.SH_spot_sim(centred = 1)

                    # Image thresholding to remove background
                    image_max = image_max - config['image']['threshold'] * np.amax(image_max)
                    image_max[image_max < 0] = 0
                    self.image.emit(image_max)

                    # Append image to list
                    self.data.append(image_max)
                    self.cent_x.append(self.spot_cent_x)
                    self.cent_y.append(self.spot_cent_y)

                    # Apply lowest voltage
                    voltages[i] = config['DM']['vol_min']

                    # Send values vector to mirror
                    self.mirror.Send(voltages)

                    # Wait for DM to settle
                    time.sleep(config['DM']['settling_time'])

                    # Acquire S-H spot image and display
                    # image_min = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)
                    spot_img = SpotSim(self.SB_settings)
                    image_min, self.spot_cent_x, self.spot_cent_y = spot_img.SH_spot_sim(centred = 1)

                    # Image thresholding to remove background
                    image_min = image_min - config['image']['threshold'] * np.amax(image_min)
                    image_min[image_min < 0] = 0
                    self.image.emit(image_min)

                    # Append image to list
                    self.data.append(image_min)
                    self.cent_x.append(self.spot_cent_x)
                    self.cent_y.append(self.spot_cent_y)

                    # Set actuator back to bias voltage
                    voltages[i] = config['DM']['vol_bias']
                else:

                    self.done.emit()
            print('Actuator calibration process finished...')

            prev2 = time.perf_counter()
            print('Time for calibration image acquisition process is:', (prev2 - prev1))

            # Reset mirror
            self.mirror.Reset()

            # Calculate S-H spot centroids for each image in data list to get slopes
            if self.calc_cent:

                print('Centroid calculation process started...')
                self.act_cent_coord, self.act_cent_coord_x, self.act_cent_coord_y, self.slope_x, self.slope_y = \
                    acq_centroid(self.SB_settings, self.cent_x, self.cent_y, self.data)
                print('Centroid calculation process finished...')
            else:

                self.done.emit()

            # Fill influence function matrix with acquired slopes
            if self.calc_inf:
                
                print('Influence function calculation process started...')
                for i in range(config['DM']['actuator_num']):

                    self.inf_matrix_slopes[:self.SB_settings['act_ref_cent_num'], i] = \
                        (self.slope_x[2 * i] - self.slope_x[2 * i + 1]) / (config['DM']['vol_max'] - config['DM']['vol_min'])
                    self.inf_matrix_slopes[self.SB_settings['act_ref_cent_num']:, i] = \
                        (self.slope_y[2 * i] - self.slope_y[2 * i + 1]) / (config['DM']['vol_max'] - config['DM']['vol_min'])
                print('Influence function calculation process finished...')  
            else:

                self.done.emit()

            prev3 = time.perf_counter()
            print('Time for entire calibration process is:', (prev3 - prev1))
            
            print('Influence function is:', self.inf_matrix_slopes)

            """
            Returns deformable mirror calibration information into self.mirror_info
            """ 
            if self.log:

                self.mirror_info['calib_spots'] = self.data
                self.mirror_info['inf_matrix_slopes'] = self.inf_matrix_slopes

                self.info.emit(self.mirror_info)
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
