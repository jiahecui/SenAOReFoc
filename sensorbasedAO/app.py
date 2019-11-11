import logging
import sys
import os
import imageio
import argparse
import math
import time
import PIL.Image
import numpy as np

from datetime import datetime

# from alpao import asdk
from ximea import xiapi

import log
from config import config

logger = log.get_logger(__name__)

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

    def register_SB(self):
        """
        Registers initial search block geometry
        """
        pixels_per_lenslet = int(self.lenslet_pitch // self.pixel_size)

        if pixels_per_lenslet % 2 == 0:
            pixels_per_lenslet += 1

        # Get search block radius and diameter
        self.SB_rad = pixels_per_lenslet // 2
        self.SB_diam = self.SB_rad * 2

    def make_reference_SB(self):
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

        # Display actual search blocks and reference centroids
        img = PIL.Image.fromarray(self.SB_layer_2D, 'L')
        img.show()

    def display_reference_SB(self):
        """
        Displays initial search blocks and reference search blocks on screen
        """

    def get_SB_geometry(self):
        """
        Calculates search block geometry from given number of spots across diameter
        """
        # Clear search block layer
        self.SB_layer_2D = np.zeros([self.sensor_width, self.sensor_height], dtype='uint8')
        self.SB_layer_1D = np.ravel(self.SB_layer_2D)

        # Initialise list of 1D coords of reference centroids within pupil diameter
        self.act_ref_cent_coord = []

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

        else:

            for j in self.ref_cent_y:
                for i in self.ref_cent_x:
                    if ((np.sqrt(((abs((i + 1 - (self.sensor_width // 2 - self.SB_rad))) + self.SB_rad) * self.pixel_size) ** 2 + \
                        ((abs((j + 1 - (self.sensor_height // 2 - self.SB_rad))) + self.SB_rad) * self.pixel_size) ** 2)) < self.pupil_rad):
                        self.act_ref_cent_coord.append(j * self.sensor_width + i)

        # Set actual search block reference centroids
        self.act_ref_cent_coord = np.array(self.act_ref_cent_coord)
        self.act_ref_cent_num = len(self.act_ref_cent_coord)
        self.SB_layer_1D[self.act_ref_cent_coord] = self.outline_int
        self.SB_layer_2D = np.reshape(self.SB_layer_1D, (self.sensor_height, self.sensor_width))
        print("Number of search blocks within pupil is: {}".format(self.act_ref_cent_num))

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
            self.SB_layer_2D[act_ref_row_top_outline[i], act_ref_cent_offset_top_x[row_top_indices[i]] :\
                act_ref_cent_offset_top_x[row_top_indices[i + 1] - 1] + self.SB_diam] = self.outline_int

        for i in range(rows // 2, rows):
            self.SB_layer_2D[act_ref_row_bottom_outline[i], act_ref_cent_offset_bottom_x[row_bottom_indices[i]] :\
                act_ref_cent_offset_bottom_x[row_bottom_indices[i] + row_bottom_counts[i]- 1] + self.SB_diam] = self.outline_int    

        # Outline columns of actual search blocks
        self.index_count = 0
        
        for i in range(columns // 2 + 1):

            if i == 0:
                self.SB_layer_2D[act_ref_cent_offset_top_x[self.index_count] : act_ref_cent_offset_top_x[self.index_count + \
                    column_left_counts[i] - 1] + self.SB_diam, act_ref_column_left_outline[i]] = self.outline_int
            else:
                self.index_count += column_left_counts[i - 1] 
                self.SB_layer_2D[act_ref_cent_offset_top_x[self.index_count] : act_ref_cent_offset_top_x[self.index_count + \
                    column_left_counts[i] - 1] + self.SB_diam, act_ref_column_left_outline[i]] = self.outline_int

        for i in range(columns // 2, columns):

            if i == columns // 2:
                self.SB_layer_2D[act_ref_cent_offset_right_x[self.index_count] - self.SB_diam : act_ref_cent_offset_right_x[self.index_count \
                    + column_right_counts[i] - 1], act_ref_column_right_outline[i]] = self.outline_int
            else:
                self.index_count += column_left_counts[i - 1] 
                self.SB_layer_2D[act_ref_cent_offset_right_x[self.index_count] - self.SB_diam : act_ref_cent_offset_right_x[self.index_count \
                    + column_right_counts[i] - 1], act_ref_column_right_outline[i]] = self.outline_int

        # Draw pupil circle on search block layer
        plot_point_num = int(self.pupil_rad * 2 // self.pixel_size * 10)

        theta = np.linspace(0, 2 * math.pi, plot_point_num)
        rho = self.pupil_rad // self.pixel_size

        if (self.spots_across_diam % 2 == 0 and self.sensor_width % 2 == 0) or \
            (self.spots_across_diam % 2 == 1 and self.sensor_width % 2 == 1):

            x = (rho * np.cos(theta)).astype(int) + self.sensor_width // 2
            y = (rho * np.sin(theta)).astype(int) + self.sensor_width // 2

        else:

            x = (rho * np.cos(theta)).astype(int) + self.sensor_width // 2 - self.SB_rad
            y = (rho * np.sin(theta)).astype(int) + self.sensor_width // 2 - self.SB_rad

        self.SB_layer_2D[x, y] = self.outline_int

        # Display actual search blocks and reference centroids
        img = PIL.Image.fromarray(self.SB_layer_2D, 'L')
        img.show()

    def get_SB_info(self):
        """
        Returns search block information
        """
        self.SB_info['SB_rad'] = self.SB_rad
        self.SB_info['SB_diam'] = self.SB_diam  
        self.SB_info['SB_across_width'] = self.SB_across_width
        self.SB_info['SB_across_height'] = self.SB_across_height
        self.SB_info['ref_cent_x'] = self.ref_cent_x
        self.SB_info['ref_cent_y'] = self.ref_cent_y
        self.SB_info['ref_cent_coord'] = self.ref_cent_coord
        self.SB_info['pupil_rad'] = self.pupil_rad
        self.SB_info['act_ref_cent_coord'] = self.act_ref_cent_coord
        self.SB_info['act_ref_cent_num'] = self.act_ref_cent_num

        return self.SB_info


class Centroiding():
    """
    Calculates centroids of S-H spots on camera sensor
    """
    def __init__(self, device, settings):

        # Get search block settings
        self.SB_settings = settings

        # Get camera instance
        self.camera = device

    def start_cam_com(self):
        self.camera.open_device_by_SN(config['camera']['SN'])
        
        return self.camera.is_isexist()

    def set_cam_settings(self):
        self.camera.set_imgdataformat(config['camera']['dataformat'])

        if config['camera']['auto_gain']:
            self.camera.is_aeag()
            self.camera.set_exp_priority(config['camera']['exp_priority'])
            self.camera.set_ae_max_limit(config['camera']['ae_max_limit'])
        else:
            self.camera.set_exposure(config['camera']['exposure'])
        
        self.camera.set_trigger_source(config['camera']['trg_source'])

        if config['camera']['burst_mode']:
            self.camera.set_trigger_selector('XI_TRG_SEL_FRAME_BURST_START')
            self.camera.set_acq_frame_burst_count(config['camera']['burst_frames'])
        else:
            self.camera.set_trigger_selector('XI_TRG_SEL_FRAME_START')

        self.camera.set_acq_timing_mode(config['camera']['acq_timing_mode'])

           
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
    logger.info('Started sensorbased AO app')

    SB = Setup_SB(debug=False)
    SB.register_SB()
    SB.make_reference_SB()
    SB.get_SB_geometry()
    SB_info = SB.get_SB_info()
    cam = xiapi.Camera()
    Cent = Centroiding(cam, SB_info)
    try:
        Cent.start_cam_com()
    except:
        logger.warning('Error loading camera')
    # Cent.set_cam_settings()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sensorbased AO gui')
    parser.add_argument("-d", "--debug", help='debug mode',
                        action="store_true")
    args = parser.parse_args()

    if args.debug:  
        debug()
    else:
        main()