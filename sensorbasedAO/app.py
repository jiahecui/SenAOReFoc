import logging
import sys
import os
import imageio
import argparse
import PIL.Image
import numpy as np

from datetime import datetime

from ximea import xiapi
# from alpao import asdk

import log
from config import config

logger = log.get_logger()

class Setup_SB():
    """
    Sets up reference search blocks
    """
    def __init__(self, debug=False):

        # Get lenslet parameters
        self.lenslet_pitch = config['lenslet']['lenslet_pitch']

        # Get camera parameters
        self.pixel_size = config['camera']['pixel_size']
        self.sensor_width = config['camera']['sensor_width']
        self.sensor_height = config['camera']['sensor_height']

        # Get search block parameters
        self.outline_int = config['search_block']['outline_int']
        self.spots_across_diam = config['search_block']['spots_across_diam']

        # Initialise search block information parameter
        self.SB_info = {}

        # Initialise search block layer
        self.SB_layer_2D = np.zeros([self.sensor_width, self.sensor_height],dtype='uint8')
        self.SB_layer_1D = np.ravel(self.SB_layer_2D)

    def register_SB(self):
        """
        Registers initial search block geometry
        """
        pixels_per_lenslet = int(self.lenslet_pitch // self.pixel_size)

        if pixels_per_lenslet % 2 == 0:
            pixels_per_lenslet += 1

        # Get search block radius
        self.SB_rad = pixels_per_lenslet // 2
        
        # Store search block information
        self.SB_info['pixels_per_lenslet'] = pixels_per_lenslet
        self.SB_info['SB_rad'] = self.SB_rad

    def make_reference_SB(self):
        """
        Makes search block layer, outlines search blocks, and creates initial reference centroids
        """
        # Get search block diameter
        self.SB_diam = self.SB_rad * 2

        # Get number of search blocks across both sensor dimensions
        if self.sensor_width % self.SB_diam == 0:
            SB_across_width = self.sensor_width // self.SB_diam - 1
        else:
            SB_across_width = self.sensor_width // self.SB_diam

        if self.sensor_height % self.SB_diam == 0:
            SB_across_height = self.sensor_height // self.SB_diam - 1
        else:
            SB_across_height = self.sensor_height // self.SB_diam

        # Get initial search block layer offset relative to sensor (top left corner)
        SB_offset_x = (self.sensor_width - (SB_across_width * self.SB_diam + 1)) // 2
        SB_offset_y = (self.sensor_height - (SB_across_height * self.SB_diam + 1)) // 2

        # Get last pixel of initial search block layer (bottom right corner)
        SB_final_x = self.sensor_width - SB_offset_x
        SB_final_y = self.sensor_height - SB_offset_y

        # Get reference centroids
        ref_cent_offset = self.SB_rad
        self.ref_cent_x = np.arange(SB_offset_x + ref_cent_offset, SB_final_x, self.SB_diam)
        self.ref_cent_y = np.arange(SB_offset_y + ref_cent_offset, SB_final_y, self.SB_diam)

        # Get 1D coords of reference centroids 
        self.ref_cent_coord = np.zeros(SB_across_width * SB_across_height, dtype=int)
        for i in np.arange(0, SB_across_height, 1):
            for j in np.arange(0, SB_across_width, 1):
                self.ref_cent_coord[i * SB_across_width + j] = self.ref_cent_y[i] * \
                    self.sensor_width + self.ref_cent_x[j]

        # Set reference centroids
        self.SB_layer_1D[self.ref_cent_coord] = self.outline_int
        self.SB_layer_2D = np.reshape(self.SB_layer_1D,(self.sensor_height, self.sensor_width))
        
        # Get arrays for outlining search blocks
        ref_row_outline = np.arange(SB_offset_y, self.sensor_height - SB_offset_y, self.SB_diam)
        ref_column_outline = np.arange(SB_offset_x, self.sensor_width - SB_offset_x, self.SB_diam)

        # Outline search blocks
        self.SB_layer_2D[ref_row_outline, SB_offset_x : (self.sensor_width - SB_offset_x)] = self.outline_int
        self.SB_layer_2D[SB_offset_y : (self.sensor_height - SB_offset_y), ref_column_outline] = self.outline_int
        
        # Store search block information
        self.SB_info['SB_diam'] = self.SB_diam
        self.SB_info['SB_across_width'] = SB_across_width
        self.SB_info['SB_across_height'] = SB_across_height
        self.SB_info['SB_offset_x'] = SB_offset_x
        self.SB_info['SB_offset_y'] = SB_offset_y
        self.SB_info['SB_final_x'] = SB_final_x
        self.SB_info['SB_final_y'] = SB_final_y
        self.SB_info['ref_cent_x'] = self.ref_cent_x
        self.SB_info['ref_cent_y'] = self.ref_cent_y
        self.SB_info['ref_cent_coord'] = self.ref_cent_coord
        self.SB_info['ref_row_outline'] = ref_row_outline
        self.SB_info['ref_column_outline'] = ref_column_outline
        
    def display_reference_SB(self):
        """
        Displays initial search blocks and reference search blocks on screen
        """

    def get_SB_geometry(self):
        """
        Calculates search block geometry from given number of spots across diameter
        """
        # Clear search block layer
        self.SB_layer_2D = np.zeros([self.sensor_width, self.sensor_height],dtype='uint8')
        self.SB_layer_1D = np.ravel(self.SB_layer_2D)

        # Get pupil diameter for given number of spots
        spots_along_diag = self.SB_info['SB_across_width']

        # Initialise list of 1D coords of reference centroids within pupil diameter
        self.act_ref_cent_coord = []

        try:
            self.spots_across_diam > 2 and self.spots_across_diam < spots_along_diag
        except ValueError as ex:
            ex_type = sys.exc_info()
            logger.error('Number of spots across diameter is out of bounds.')
            print('Exception type: %s ' % ex_type.__name__)

        if (self.spots_across_diam < 2 or self.spots_across_diam > spots_along_diag):
            print('Number of spots across diameter is out of bounds.')

        for i in range(spots_along_diag // 2 - 2):
            pupil_diam_temp_max = int(np.sqrt(2) * (self.sensor_width // 2 - \
                self.ref_cent_x[i] + self.SB_rad) * self.pixel_size)
            pupil_diam_temp_min = int(np.sqrt(2) * (self.sensor_width // 2 - \
                self.ref_cent_x[i+1] + self.SB_rad) * self.pixel_size)

            if self.spots_across_diam % 2 == 1:
                pixel_edge = int((self.SB_diam * (self.spots_across_diam // 2) + self.SB_rad) * self.pixel_size)
            else:
                pixel_edge = int(self.SB_diam * (self.spots_across_diam // 2) * self.pixel_size)
            
            if pupil_diam_temp_max > pixel_edge and pupil_diam_temp_min < pixel_edge:
                self.pupil_diam = pupil_diam_temp_max
                break

        # Get reference centroids within pupil diameter
        for j in self.ref_cent_y:
            for i in self.ref_cent_x:
                if ((np.sqrt(((abs((i - self.sensor_width // 2)) + self.SB_rad) * self.pixel_size) ** 2 + \
                    ((abs((j - self.sensor_height // 2)) +self.SB_rad) * self.pixel_size) ** 2)) < self.pupil_diam):
                    self.act_ref_cent_coord.append(j * self.sensor_width + i)

        # Set actual search block reference centroids
        self.act_ref_cent_coord = np.array(self.act_ref_cent_coord)
        self.SB_layer_1D[self.ref_cent_coord] = self.outline_int
        self.SB_layer_2D = np.reshape(self.SB_layer_1D,(self.sensor_height, self.sensor_width))

        # Get 1D coord offset of each actual search block
        act_ref_cent_offset_top_coord = self.act_ref_cent_coord - self.SB_rad * self.sensor_width - self.SB_rad
        act_ref_cent_offset_bottom_coord = self.act_ref_cent_coord + self.SB_rad * self.sensor_width - self.SB_rad
        act_ref_cent_offset_right_coord = self.act_ref_cent_coord - self.SB_rad * self.sensor_width + self.SB_rad

        # Get 2D coord offset of each actual search block
        act_ref_cent_offset_top_y = act_ref_cent_offset_top_coord // self.sensor_width
        act_ref_cent_offset_top_x = act_ref_cent_offset_top_coord - act_ref_cent_offset_top_y * self.sensor_width
        act_ref_cent_offset_bottom_y = act_ref_cent_offset_bottom_coord // self.sensor_width
        act_ref_cent_offset_bottom_x = act_ref_cent_offset_bottom_coord - act_ref_cent_offset_bottom_y * self.sensor_width
        act_ref_cent_offset_right_y = act_ref_cent_offset_right_coord // self.sensor_width
        act_ref_cent_offset_right_x = act_ref_cent_offset_right_coord - act_ref_cent_offset_right_y * self.sensor_width

        # Get parameters for outlining actual search blocks
        act_ref_row_outline, row_indices, row_counts =\
            np.unique(act_ref_cent_offset_top_y, return_index = True, return_counts = True)
        act_ref_column_outline, column_indices, column_counts =\
            np.unique(act_ref_cent_offset_top_x, return_index = True, return_counts = True)

        # Outline search blocks
        rows = len(act_ref_row_outline)
        columns = len(act_ref_column_outline)

        for i in range(rows // 2):
            self.SB_layer_2D[act_ref_row_outline[i], act_ref_cent_offset_top_x[row_indices[i]] :\
                act_ref_cent_offset_top_x[row_indices[i + 1] - 1] + self.SB_diam] = self.outline_int

        for i in range(rows // 2, rows):
            if i < (rows - 1):
                self.SB_layer_2D[act_ref_row_outline[i], act_ref_cent_offset_bottom_x[row_indices[i]] :\
                    act_ref_cent_offset_bottom_x[row_indices[i + 1] - 1] + self.SB_diam] = self.outline_int
            else:
                self.SB_layer_2D[act_ref_row_outline[i], act_ref_cent_offset_bottom_x[row_indices[i]] :\
                    act_ref_cent_offset_bottom_x[row_indices[i] - 1 + row_counts[i]] + self.SB_diam] = self.outline_int

        index_count_1 = 0
        index_count_2 = 0

        for i in range(columns // 2):

            index_count_1 = index_count_2
            index_count_2 = index_count_1 + column_counts[i] - 1

            self.SB_layer_2D[act_ref_cent_offset_top_x[index_count_1] : act_ref_cent_offset_top_x[index_count_2] \
                + self.SB_diam, act_ref_column_outline[i]] = self.outline_int

        for i in range(columns // 2, columns):

            index_count_1 = index_count_2
            index_count_2 = index_count_1 + column_counts[i] - 1

            self.SB_layer_2D[act_ref_cent_offset_right_x[index_count_1] : act_ref_cent_offset_right_x[index_count_2] \
                + self.SB_diam, act_ref_column_outline[i]] = self.outline_int

        # Store search block information
        self.SB_info['pupil_diam'] = self.pupil_diam


def debug():
    logger.setLevel(logging.DEBUG)
    handler_stream = logging.StreamHandler()
    handler_stream.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_stream)
    logger.info('Started sensor-based AO app in debug mode')

    app = App(debug=True)

    sys.exit(app.exec_())

def main():
    logger.setLevel(logging.DEBUG)
    handler_stream = logging.StreamHandler()  
    handler_stream.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_stream)
    logger.info('Started sensor-based AO app')

    app = Setup_SB(debug=False)
    app.register_SB()
    app.make_reference_SB()
    app.get_SB_geometry()

    img = PIL.Image.fromarray(app.SB_layer_2D, 'L')
    img.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sensor-based AO gui')
    parser.add_argument("-d", "--debug", help='debug mode',
                        action="store_true")
    args = parser.parse_args()

    if args.debug:
        debug()
    else:
        main()