from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import logging
import sys
import os
import argparse
import math
import time
import PIL.Image
import numpy as np

import log
from config import config

logger = log.get_logger(__name__)

class Setup_SB(QObject):
    """
    Sets up reference search block geometry
    """
    start = Signal()
    done = Signal()
    error = Signal(object)
    layer = Signal(object)
    message = Signal(object)
    info = Signal(object)

    def __init__(self, mirror):

        # Get mirror instance
        self.mirror = mirror

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']

        # Get lenslet parameters
        self.lenslet_pitch = config['lenslet']['lenslet_pitch']

        # Get search block layer parameters
        self.pixel_size = config['camera']['pixel_size']
        self.sensor_width = int(config['camera']['sensor_width'] // config['camera']['bin_factor'])
        self.sensor_height = int(config['camera']['sensor_height'] // config['camera']['bin_factor'])

        # Get search block parameter
        self.outline_int = config['search_block']['outline_int']

        # Get pupil diameter
        if config['DM']['DM_num'] == 0:
            self.pupil_diam = config['search_block']['pupil_diam_0'] * 1e3
        elif config['DM']['DM_num'] == 1:
            self.pupil_diam = config['search_block']['pupil_diam_1'] * 1e3
        self.pupil_rad = self.pupil_diam / 2

        # Initialise search block information parameter
        self.SB_info = {}

        # Initialise search block layer
        self.SB_layer_2D = np.zeros([self.sensor_height, self.sensor_width], dtype = 'uint8')

        super().__init__()

    @Slot(object)
    def run(self):
        try:
            # Set process flags
            self.register = True
            self.calculate = True
            self.log = True

            # Start thread
            self.start.emit()

            # Get search block radius and diameter
            self.SB_diam = int(self.lenslet_pitch // self.pixel_size)
            self.SB_rad = self.SB_diam / 2
            
            """
            Makes search block layer, outlines search blocks, and creates initial reference centroids
            """
            # Get number of search blocks across both sensor dimensions
            if self.sensor_width % self.SB_diam == 0:
                self.SB_across_width = int(self.sensor_width // self.SB_diam - 1)
            else:
                self.SB_across_width = int(self.sensor_width // self.SB_diam)

            if self.sensor_height % self.SB_diam == 0:
                self.SB_across_height = int(self.sensor_height // self.SB_diam - 1)
            else:
                self.SB_across_height = int(self.sensor_height // self.SB_diam)

            # print("Number of search blocks across width and height is: {} and {}".format(self.SB_across_width, self.SB_across_height))

            # Get initial search block layer offset relative to sensor (top left corner)
            SB_offset_x = (self.sensor_width - (self.SB_across_width * self.SB_diam)) / 2
            SB_offset_y = (self.sensor_height - (self.SB_across_height * self.SB_diam)) / 2

            # Get last pixel of initial search block layer (bottom right corner)
            SB_final_x = SB_offset_x + self.SB_across_width * self.SB_diam
            SB_final_y = SB_offset_y + self.SB_across_height * self.SB_diam  

            # print('Offset_x: {}, offset_y: {}, final_x: {}, final_y: {}'.format(SB_offset_x, SB_offset_y, SB_final_x, SB_final_y))

            # Get reference centroids
            self.ref_cent_x = np.arange(SB_offset_x + self.SB_rad, SB_final_x, self.SB_diam)
            self.ref_cent_y = np.arange(SB_offset_y + self.SB_rad, SB_final_y, self.SB_diam)

            # print('Ref_cent_x: {}, ref_cent_y: {}'.format(self.ref_cent_x, self.ref_cent_y))

            # Get 1D coords of reference centroids 
            self.ref_cent_coord = np.zeros(self.SB_across_width * self.SB_across_height)
          
            for i in range(self.SB_across_height):
                for j in range(self.SB_across_width):
                    self.ref_cent_coord[i * self.SB_across_width + j] = self.ref_cent_y[i].astype(int) * \
                        self.sensor_width + self.ref_cent_x[j].astype(int)
        
            # Get arrays for outlining search blocks
            ref_row_outline = np.arange(SB_offset_y, SB_final_y + 1, self.SB_diam).astype(int)
            ref_column_outline = np.arange(SB_offset_x, SB_final_x + 1, self.SB_diam).astype(int)

            # Outline search blocks
            self.SB_layer_2D[ref_row_outline, int(SB_offset_x) : int(SB_final_x)] = self.outline_int
            self.SB_layer_2D[int(SB_offset_y) : int(SB_final_y), ref_column_outline] = self.outline_int

            # Display search blocks and reference centroids
            if self.register:

                self.layer.emit(self.SB_layer_2D)
            else:

                self.done.emit()

            """
            Calculates search block geometry from given number of spots across diameter
            """
            # Clear search block layer
            self.SB_layer_2D = np.zeros([self.sensor_height, self.sensor_width], dtype = 'uint8')

            # Initialise list of 1D and 2D coords of reference centroids within pupil diameter
            self.act_ref_cent_coord, self.act_ref_cent_coord_x, self.act_ref_cent_coord_y = ([] for i in range(3))

            # Get number of spots within pupil diameter
            self.spots_across_diam = self.pupil_diam // self.lenslet_pitch

            # Get actual search block reference centroids within pupil diameter
            if (self.spots_across_diam % 2 == 0 and self.SB_across_width % 2 == 0) or \
                (self.spots_across_diam % 2 == 1 and self.SB_across_width % 2 == 1):

                for j in self.ref_cent_y:
                    for i in self.ref_cent_x:
                        if ((np.sqrt(((abs((i - self.sensor_width // 2)) + self.SB_rad) * self.pixel_size) ** 2 + \
                            ((abs((j - self.sensor_height // 2)) + self.SB_rad) * self.pixel_size) ** 2)) <= self.pupil_rad):
                            self.act_ref_cent_coord.append(int(j) * self.sensor_width + int(i))
                            self.act_ref_cent_coord_x.append(i)
                            self.act_ref_cent_coord_y.append(j)

            else:

                for j in self.ref_cent_y:
                    for i in self.ref_cent_x:
                        if ((np.sqrt(((abs((i - (self.sensor_width // 2 - self.SB_rad))) + self.SB_rad) * self.pixel_size) ** 2 + \
                            ((abs((j - (self.sensor_height // 2 - self.SB_rad))) + self.SB_rad) * self.pixel_size) ** 2)) <= self.pupil_rad):
                            self.act_ref_cent_coord.append(int(j) * self.sensor_width + int(i))
                            self.act_ref_cent_coord_x.append(i)
                            self.act_ref_cent_coord_y.append(j)

            (self.act_ref_cent_coord, self.act_ref_cent_coord_x, self.act_ref_cent_coord_y) = \
                map(np.array, (self.act_ref_cent_coord, self.act_ref_cent_coord_x, self.act_ref_cent_coord_y))
            self.act_ref_cent_num = len(self.act_ref_cent_coord)

            # Shift search blocks to the centre if the number of spots across diameter and the number of search blocks across sensor width 
            # aren't both odd or both even
            if (self.spots_across_diam % 2 == 0 and self.SB_across_width % 2 == 1) or \
                (self.spots_across_diam % 2 == 1 and self.SB_across_width % 2 == 0):
                self.act_ref_cent_coord_x += int(self.SB_rad)
                self.act_ref_cent_coord_y += int(self.SB_rad)
                self.act_ref_cent_coord += int(self.SB_rad) * self.sensor_width + int(self.SB_rad)

            print("Number of search blocks within pupil is: {}".format(self.act_ref_cent_num))
        
            # Draw actual search blocks on search block layer
            for i in range(self.act_ref_cent_num):

                # If odd number of pixels in a search block
                if self.SB_diam % 2 == 1:
                    self.odd_pix = 1
                    # Outline top
                    self.SB_layer_2D[int(self.act_ref_cent_coord_y[i] - self.SB_rad), \
                        int(self.act_ref_cent_coord_x[i] - self.SB_rad) : int(self.act_ref_cent_coord_x[i] + self.SB_rad)] = self.outline_int
                    # Outline bottom
                    self.SB_layer_2D[int(self.act_ref_cent_coord_y[i] + self.SB_rad), \
                        int(self.act_ref_cent_coord_x[i] - self.SB_rad) : int(self.act_ref_cent_coord_x[i] + self.SB_rad)] = self.outline_int
                    # Outline left
                    self.SB_layer_2D[int(self.act_ref_cent_coord_y[i] - self.SB_rad) : int(self.act_ref_cent_coord_y[i] + self.SB_rad), \
                        int(self.act_ref_cent_coord_x[i] - self.SB_rad)] = self.outline_int
                    # Outline right
                    self.SB_layer_2D[int(self.act_ref_cent_coord_y[i] - self.SB_rad) : int(self.act_ref_cent_coord_y[i] + self.SB_rad), \
                        int(self.act_ref_cent_coord_x[i] + self.SB_rad)] = self.outline_int

                # If even number of pixels in a search block
                elif self.SB_diam % 2 == 0:
                    self.odd_pix = 0
                    # Outline top
                    self.SB_layer_2D[int(self.act_ref_cent_coord_y[i] - self.SB_rad + 1), \
                        int(self.act_ref_cent_coord_x[i] - self.SB_rad + 1) : int(self.act_ref_cent_coord_x[i] + self.SB_rad)] = self.outline_int
                    # Outline bottom
                    self.SB_layer_2D[int(self.act_ref_cent_coord_y[i] + self.SB_rad), \
                        int(self.act_ref_cent_coord_x[i] - self.SB_rad + 1) : int(self.act_ref_cent_coord_x[i] + self.SB_rad)] = self.outline_int
                    # Outline left
                    self.SB_layer_2D[int(self.act_ref_cent_coord_y[i] - self.SB_rad + 1) : int(self.act_ref_cent_coord_y[i] + self.SB_rad), \
                        int(self.act_ref_cent_coord_x[i] - self.SB_rad + 1)] = self.outline_int
                    # Outline right
                    self.SB_layer_2D[int(self.act_ref_cent_coord_y[i] - self.SB_rad + 1) : int(self.act_ref_cent_coord_y[i] + self.SB_rad), \
                        int(self.act_ref_cent_coord_x[i] + self.SB_rad)] = self.outline_int

            # Draw pupil circle on search block layer
            plot_point_num = int(self.pupil_diam // self.pixel_size * 10)

            theta = np.linspace(0, 2 * np.pi, plot_point_num)
            rho = self.pupil_rad // self.pixel_size

            x = (rho * np.cos(theta) + self.sensor_width // 2).astype(int)
            y = (rho * np.sin(theta) + self.sensor_width // 2).astype(int)
            
            self.SB_layer_2D[x, y] = self.outline_int

            # Display actual search blocks
            if self.calculate:

                self.layer.emit(self.SB_layer_2D)
                self.message.emit('Search block geometry initialised.')
            else:

                self.done.emit()

            # Get actual search block coordinates
            self.act_SB_coord = np.nonzero(np.ravel(self.SB_layer_2D))
            self.act_SB_coord = np.array(self.act_SB_coord)

            # Determine whether to exercise DM upon initialisation
            if config['DM']['exercise']:

                print('DM exercise started...')
                   
                for i in range(config['DM']['exercise_num']):

                    # Initialise deformable mirror voltage array
                    voltages = np.ravel(1.0 * (0.5 - np.random.rand(self.actuator_num, 1)))

                    # Send voltages to exercise DM
                    self.mirror.Send(voltages)

                    # Wait for DM to settle
                    time.sleep(config['DM']['settling_time'])
                    
                # Reset DM
                self.mirror.Reset()

                print('DM exercise finished.')

            """
            Returns search block information
            """
            if self.log:

                self.SB_info['pixel_size'] = self.pixel_size  # float
                self.SB_info['sensor_width'] = self.sensor_width  # int after binning
                self.SB_info['sensor_height'] = self.sensor_height  # int after binning
                self.SB_info['SB_rad'] = self.SB_rad  # float
                self.SB_info['SB_diam'] = self.SB_diam  # int
                self.SB_info['SB_across_width'] = self.SB_across_width  # int
                self.SB_info['SB_across_height'] = self.SB_across_height  # int
                self.SB_info['spots_across_diam'] = self.spots_across_diam  # int
                self.SB_info['act_ref_cent_num'] = self.act_ref_cent_num
                self.SB_info['odd_pix'] = self.odd_pix  # flag for whether odd number of pixels in one search block
                self.SB_info['act_ref_cent_coord'] = self.act_ref_cent_coord  # int - for displaying
                self.SB_info['act_ref_cent_coord_x'] = self.act_ref_cent_coord_x  # float - actual reference centroid positions, not whole pixels
                self.SB_info['act_ref_cent_coord_y'] = self.act_ref_cent_coord_y  # float - actual reference centroid positions, not whole pixels
                self.SB_info['act_SB_coord'] = self.act_SB_coord  # int - for displaying
                self.SB_info['act_SB_offset_x'] = 0  # int - for storing x_offset relative to centre of sensor
                self.SB_info['act_SB_offset_y'] = 0  # int - for storing y_offset relative to centre of sensor

                self.info.emit(self.SB_info)
            else:

                self.done.emit()

            # Finished setting up search block geometry
            self.done.emit()

        except Exception as e:
            self.error.emit(e)
            raise

    @Slot()
    def stop(self):
        self.register = False
        self.calculate = False
        self.log = False
