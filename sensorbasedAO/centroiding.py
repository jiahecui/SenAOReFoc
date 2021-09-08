from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import sys
import os
import argparse
import time
import h5py
import numpy as np

import log
from config import config
from HDF5_dset import dset_append, get_dset
from image_acquisition import acq_image
from centroid_acquisition import acq_centroid

logger = log.get_logger(__name__)

class Centroiding(QObject):
    """
    Calculates centroids of S-H spots on camera sensor for system aberration calibration
    """
    start = Signal()
    write = Signal()
    done = Signal()
    error = Signal(object)
    layer = Signal(object)
    message = Signal(object)
    SB_info = Signal(object)
    
    def __init__(self, sensor, mirror, settings):

        # Get search block settings
        self.SB_settings = settings

        # Get sensor instance
        self.sensor = sensor

        # Get mirror instance
        self.mirror = mirror

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']

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
            Calculate actual S-H spot centroid coordinates for system aberrations
            """
            # Create new datasets in HDF5 file to store centroiding data
            get_dset(self.SB_settings, 'centroiding_img', flag = 3)
            data_file = h5py.File('data_info.h5', 'a')
            data_set = data_file['centroiding_img']

            self.message.emit('\nSystem aberration calibration process started...')

            # Initialise search block layer and display search blocks
            SB_layer_2D = np.zeros([self.SB_settings['sensor_height'], self.SB_settings['sensor_width']])
            SB_layer_2D_temp = SB_layer_2D.copy()
            SB_layer_2D_temp.ravel()[self.SB_settings['act_SB_coord']] = config['search_block']['outline_int']

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            # Select system aberration calibration mode
            if config['sys_calib']['sys_calib_mode'] == 1:
                voltages = h5py.File('exec_files/flat_volts_1.mat','r').get('flat_volts')
                voltages = np.ravel(np.array(voltages))
                self.mirror.Send(voltages)

            # Acquire S-H spot image
            cent_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
            cent_image = np.mean(cent_image_stack, axis = 2)
            
            # Image thresholding to remove background
            cent_image = cent_image - config['image']['threshold'] * np.amax(cent_image)
            cent_image[cent_image < 0] = 0
            SB_layer_2D_temp += cent_image

            self.layer.emit(SB_layer_2D_temp)

            # Append image to list
            dset_append(data_set, 'real_cent_img', cent_image)

            # Calculate centroids for S-H spots
            if self.calc_cent:
                
                # Acquire centroid information
                act_ref_cent_coord, act_ref_cent_coord_x, act_ref_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 0)
                act_ref_cent_coord, act_ref_cent_coord_x, act_ref_cent_coord_y = map(np.asarray, [act_ref_cent_coord, act_ref_cent_coord_x, act_ref_cent_coord_y])

                # Draw actual S-H spot centroids
                SB_layer_2D_temp.ravel()[act_ref_cent_coord.astype(int)] = 0
                self.layer.emit(SB_layer_2D_temp)

                # Take tip\tilt off
                act_ref_cent_coord_x -= np.mean(slope_x)
                act_ref_cent_coord_y -= np.mean(slope_y)

                print(np.mean(slope_x), np.mean(slope_y))

                slope_x -= np.mean(slope_x)
                slope_y -= np.mean(slope_y)

                self.message.emit('\nSystem aberration calibration process finished.')
            else:

                self.done.emit()

            """
            Returns system aberration information into self.SB_info
            """ 
            if self.log:

                self.SB_settings['act_ref_cent_coord_x'] = act_ref_cent_coord_x
                self.SB_settings['act_ref_cent_coord_y'] = act_ref_cent_coord_y
                self.SB_settings['act_ref_cent_coord'] = act_ref_cent_coord
                self.SB_settings['real_spot_slope_x'] = slope_x
                self.SB_settings['real_spot_slope_y'] = slope_y

                self.SB_info.emit(self.SB_settings)
                self.write.emit()
            else:

                self.done.emit()
       
            # Finished calculating centroids of S-H spots
            self.done.emit()

        except Exception as e:
            self.error.emit(e)
            raise

    @Slot()
    def stop(self):
        self.calc_cent = False
        self.log = False


    