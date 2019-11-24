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
    info = Signal(object)

    def __init__(self, debug = False):

        # Get lenslet parameters
        self.lenslet_pitch = config['lenslet']['lenslet_pitch']

        # Get search block layer parameters
        self.pixel_size = config['camera']['pixel_size'] * config['camera']['bin_factor']
        self.sensor_width = config['camera']['sensor_width'] // config['camera']['bin_factor']
        self.sensor_height = config['camera']['sensor_height'] // config['camera']['bin_factor']

        # Get search block parameter
        self.outline_int = config['search_block']['outline_int']

        # Get pupil diameter
        self.pupil_diam = config['search_block']['pupil_diam'] * 1e3
        self.pupil_rad = self.pupil_diam / 2

        # Initialise search block information parameter
        self.SB_info = {}

        # Initialise search block layer
        self.SB_layer_2D = np.zeros([self.sensor_width, self.sensor_height], dtype='uint8')
        self.SB_layer_1D = np.ravel(self.SB_layer_2D)

        super().__init__()

    @Slot(object)
    def run(self):
        try:
            self.start.emit()

            """
            Registers initial search block geometry
            """
            pixels_per_lenslet = int(self.lenslet_pitch // self.pixel_size)

            if pixels_per_lenslet % 2 == 0:
                pixels_per_lenslet += 1

            # Get search block radius and diameter
            self.SB_rad = pixels_per_lenslet // 2
            self.SB_diam = self.SB_rad * 2

            """
            Makes search block layer, outlines search blocks, and creates initial reference centroids
            """
            # Get number of search blocks across both sensor dimensions
            if self.sensor_width % self.SB_diam == 0:
                self.SB_across_width = self.sensor_width // self.SB_diam - 1
            else:
                self.SB_across_width = self.sensor_width // self.SB_diam

            if self.sensor_height % self.SB_diam == 0:
                self.SB_across_height = self.sensor_height // self.SB_diam - 1
            else:
                self.SB_across_height = self.sensor_height // self.SB_diam

            print("Number of search blocks across width and height is: {} and {}".format(self.SB_across_width, self.SB_across_height))

            # Get initial search block layer offset relative to sensor (top left corner)
            SB_offset_x = (self.sensor_width - (self.SB_across_width * self.SB_diam + 1)) // 2
            SB_offset_y = (self.sensor_height - (self.SB_across_height * self.SB_diam + 1)) // 2

            # Get last pixel of initial search block layer (bottom right corner)
            SB_final_x = self.sensor_width - SB_offset_x
            SB_final_y = self.sensor_height - SB_offset_y

            # print('Offset_x: {}, offset_y: {}, final_x: {}, final_y: {}'.format(SB_offset_x, SB_offset_y, SB_final_x, SB_final_y))

            # Get reference centroids
            self.ref_cent_x = np.arange(SB_offset_x + self.SB_rad, SB_final_x - self.SB_rad, self.SB_diam)
            self.ref_cent_y = np.arange(SB_offset_y + self.SB_rad, SB_final_y - self.SB_rad, self.SB_diam)

            # Get 1D coords of reference centroids 
            self.ref_cent_coord = np.zeros(self.SB_across_width * self.SB_across_height, dtype=int)
            
            for i in range(self.SB_across_height):
                for j in range(self.SB_across_width):
                    self.ref_cent_coord[i * self.SB_across_width + j] = self.ref_cent_y[i] * \
                        self.sensor_width + self.ref_cent_x[j]

            # Set reference centroids
            self.SB_layer_1D[self.ref_cent_coord] = self.outline_int
            self.SB_layer_2D = np.reshape(self.SB_layer_1D,(self.sensor_height, self.sensor_width))
            
            # Get arrays for outlining search blocks
            ref_row_outline = np.arange(SB_offset_y, SB_final_y, self.SB_diam)
            ref_column_outline = np.arange(SB_offset_x, SB_final_x, self.SB_diam)

            # Outline search blocks
            self.SB_layer_2D[ref_row_outline, SB_offset_x : SB_final_x] = self.outline_int
            self.SB_layer_2D[SB_offset_y : SB_final_y, ref_column_outline] = self.outline_int

            # Display search blocks and reference centroids
            # im = PIL.Image.fromarray(self.SB_layer_2D, 'L')
            # im.show()
            self.layer.emit(self.SB_layer_2D)
            time.sleep(1)

            """
            Calculates search block geometry from given number of spots across diameter
            """
            # Clear search block layer
            self.SB_layer_2D = np.zeros([self.sensor_width, self.sensor_height], dtype='uint8')

            # Initialise list of 1D and 2D coords of reference centroids within pupil diameter
            self.act_ref_cent_coord = []
            self.act_ref_cent_coord_x = []
            self.act_ref_cent_coord_y = []

            # Get number of spots within pupil diameter
            self.spots_across_diam = self.pupil_diam // self.lenslet_pitch

            # Get reference centroids within pupil diameter
            if (self.spots_across_diam % 2 == 0 and self.sensor_width % 2 == 0) or \
                (self.spots_across_diam % 2 == 1 and self.sensor_width % 2 == 1):

                for j in self.ref_cent_y:
                    for i in self.ref_cent_x:
                        if ((np.sqrt(((abs((i + 1 - self.sensor_width // 2)) + self.SB_rad) * self.pixel_size) ** 2 + \
                            ((abs((j + 1 - self.sensor_height // 2)) + self.SB_rad) * self.pixel_size) ** 2)) < self.pupil_rad):
                            self.act_ref_cent_coord.append(j * self.sensor_width + i)
                            self.act_ref_cent_coord_x.append(i)
                            self.act_ref_cent_coord_y.append(j)

            else:

                for j in self.ref_cent_y:
                    for i in self.ref_cent_x:
                        if ((np.sqrt(((abs((i + 1 - (self.sensor_width // 2 - self.SB_rad))) + self.SB_rad) * self.pixel_size) ** 2 + \
                            ((abs((j + 1 - (self.sensor_height // 2 - self.SB_rad))) + self.SB_rad) * self.pixel_size) ** 2)) < self.pupil_rad):
                            self.act_ref_cent_coord.append(j * self.sensor_width + i)
                            self.act_ref_cent_coord_x.append(i)
                            self.act_ref_cent_coord_y.append(j)
                      
            # Set actual search block reference centroids
            self.act_ref_cent_coord = np.array(self.act_ref_cent_coord)
            self.act_ref_cent_coord_x = np.array(self.act_ref_cent_coord_x)
            self.act_ref_cent_coord_y = np.array(self.act_ref_cent_coord_y)
            self.act_ref_cent_num = len(self.act_ref_cent_coord)
            self.SB_layer_2D.ravel()[(self.act_ref_cent_coord - self.sensor_width // 2).astype(int)] = self.outline_int
            
            # print("Number of search blocks within pupil is: {}".format(self.act_ref_cent_num))
           
            # Get 1D coord offset of each actual search block
            act_ref_cent_offset_top_coord = self.act_ref_cent_coord - self.SB_rad * self.sensor_width - self.SB_rad
            act_ref_cent_offset_bottom_coord = self.act_ref_cent_coord + self.SB_rad * self.sensor_width - self.SB_rad
            act_ref_cent_offset_right_coord = self.act_ref_cent_coord - self.SB_rad * self.sensor_width + self.SB_rad

            # Get 2D coord offset of each actual search block
            act_ref_cent_offset_top_y = act_ref_cent_offset_top_coord // self.sensor_width
            act_ref_cent_offset_top_x = act_ref_cent_offset_top_coord % self.sensor_width
            act_ref_cent_offset_bottom_y = act_ref_cent_offset_bottom_coord // self.sensor_width
            act_ref_cent_offset_bottom_x = act_ref_cent_offset_bottom_coord % self.sensor_width
            act_ref_cent_offset_right_y = act_ref_cent_offset_right_coord // self.sensor_width
            act_ref_cent_offset_right_x = act_ref_cent_offset_right_coord % self.sensor_width

            # Get parameters for outlining actual search blocks
            act_ref_row_top_outline, row_top_indices, row_top_counts =\
                np.unique(act_ref_cent_offset_top_y, return_index = True, return_counts = True)
            act_ref_row_bottom_outline, row_bottom_indices, row_bottom_counts =\
                np.unique(act_ref_cent_offset_bottom_y, return_index = True, return_counts = True)
            act_ref_column_left_outline, column_left_indices, column_left_counts =\
                np.unique(act_ref_cent_offset_top_x, return_index = True, return_counts = True)
            act_ref_column_right_outline, column_right_indices, column_right_counts =\
                np.unique(act_ref_cent_offset_right_x, return_index = True, return_counts = True)

            # Get number of rows and columns for outlining
            rows = len(act_ref_row_top_outline)
            columns = len(act_ref_column_left_outline)

            # Outline rows of actual search blocks
            for i in range(rows // 2 + 1):
                self.SB_layer_2D[int(act_ref_row_top_outline[i]), int(act_ref_cent_offset_top_x[row_top_indices[i]]) :\
                    int(act_ref_cent_offset_top_x[row_top_indices[i + 1] - 1] + self.SB_diam)] = self.outline_int

            for i in range(rows // 2, rows):
                self.SB_layer_2D[int(act_ref_row_bottom_outline[i]), int(act_ref_cent_offset_bottom_x[row_bottom_indices[i]]) :\
                    int(act_ref_cent_offset_bottom_x[row_bottom_indices[i] + row_bottom_counts[i]- 1] + self.SB_diam)] = self.outline_int    

            # Outline columns of actual search blocks
            self.index_count = 0
            
            for i in range(columns // 2 + 1):

                if i == 0:
                    self.SB_layer_2D[int(act_ref_cent_offset_top_x[self.index_count]) : int(act_ref_cent_offset_top_x[self.index_count + \
                        column_left_counts[i] - 1] + self.SB_diam), int(act_ref_column_left_outline[i])] = self.outline_int
                else:
                    self.index_count += column_left_counts[i - 1] 
                    self.SB_layer_2D[int(act_ref_cent_offset_top_x[self.index_count]) : int(act_ref_cent_offset_top_x[self.index_count + \
                        column_left_counts[i] - 1] + self.SB_diam), int(act_ref_column_left_outline[i])] = self.outline_int

            for i in range(columns // 2, columns):

                if i == columns // 2:
                    self.SB_layer_2D[int(act_ref_cent_offset_right_x[self.index_count] - self.SB_diam) : int(act_ref_cent_offset_right_x[self.index_count \
                        + column_right_counts[i] - 1]), int(act_ref_column_right_outline[i])] = self.outline_int
                else:
                    self.index_count += column_left_counts[i - 1] 
                    self.SB_layer_2D[int(act_ref_cent_offset_right_x[self.index_count] - self.SB_diam) : int(act_ref_cent_offset_right_x[self.index_count \
                        + column_right_counts[i] - 1]), int(act_ref_column_right_outline[i])] = self.outline_int
            
            # Draw pupil circle on search block layer
            plot_point_num = int(self.pupil_rad * 2 // self.pixel_size * 10)

            theta = np.linspace(0, 2 * math.pi, plot_point_num)
            rho = self.pupil_rad // self.pixel_size

            if (self.spots_across_diam % 2 == 0 and self.sensor_width % 2 == 0) or \
                (self.spots_across_diam % 2 == 1 and self.sensor_width % 2 == 1):

                x = (rho * np.cos(theta) + self.sensor_width // 2).astype(int)
                y = (rho * np.sin(theta) + self.sensor_width // 2).astype(int)

            else:

                x = (rho * np.cos(theta) + self.sensor_width // 2 - self.SB_rad).astype(int) 
                y = (rho * np.sin(theta) + self.sensor_width // 2 - self.SB_rad).astype(int)

            self.SB_layer_2D[x, y] = self.outline_int

            # Display actual search blocks and reference centroids
            # im = PIL.Image.fromarray(self.SB_layer_2D, 'L')
            # im.show()
            self.layer.emit(self.SB_layer_2D)
            time.sleep(1)

            # Get actual search block coordinates
            self.act_SB_coord = np.nonzero(np.ravel(self.SB_layer_2D))
            self.act_SB_coord = np.array(self.act_SB_coord)

            """
            Returns search block information
            """
            self.SB_info['pixel_size'] = self.pixel_size
            self.SB_info['sensor_width'] = self.sensor_width
            self.SB_info['sensor_height'] = self.sensor_height
            self.SB_info['SB_rad'] = self.SB_rad
            self.SB_info['SB_diam'] = self.SB_diam  
            self.SB_info['SB_across_width'] = self.SB_across_width
            self.SB_info['SB_across_height'] = self.SB_across_height
            self.SB_info['act_ref_cent_coord'] = self.act_ref_cent_coord - self.sensor_width // 2
            self.SB_info['act_ref_cent_coord_x'] = self.act_ref_cent_coord_x
            self.SB_info['act_ref_cent_coord_y'] = self.act_ref_cent_coord_y
            self.SB_info['act_ref_cent_num'] = self.act_ref_cent_num
            self.SB_info['act_SB_coord'] = self.act_SB_coord

            self.info.emit(self.SB_info)

            # Finished setting up search block geometry
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)