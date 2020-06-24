from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import logging
import sys
import os
import argparse
import time
import h5py
import math
from numpy.random import seed
from numpy.random import rand
from scipy import io
from skimage.restoration import unwrap_phase
import numpy as np
import scipy as sp

import log
from config import config
from zernike_phase import zern_phase
from common import fft_spot_from_phase

logger = log.get_logger(__name__)

class ML_Dataset_Gen(QObject):
    """
    Generates dataset for ML by scanning samples with a given amount of aberration
    """
    start = Signal()
    done = Signal()
    error = Signal(object)
    message = Signal(object)
    image = Signal(object)

    def __init__(self, sensor, mirror, settings):

        # Get search block settings
        self.SB_settings = settings['SB_info']

        # Get mirror settings
        self.mirror_settings = settings['mirror_info']

        # Get AO settings
        self.AO_settings = settings['AO_info']

        # Get sensor instance
        self.sensor = sensor

        # Get mirror instance
        self.mirror = mirror

        # Generate boolean phase mask
        self.phase_rad = int(config['search_block']['pupil_diam_0'] * 1e3 / self.SB_settings['pixel_size']) // 2
        self.coord_x, self.coord_y = (np.arange(int(-self.SB_settings['sensor_width'] / 2), int(-self.SB_settings['sensor_width'] / 2) + \
            self.SB_settings['sensor_width']) for i in range(2))
        self.coord_xx, self.coord_yy = np.meshgrid(self.coord_x, self.coord_y)
        self.phase_mask = np.sqrt(self.coord_xx ** 2 + self.coord_yy ** 2) < self.phase_rad

        super().__init__()

    def aberr_gen(self):
        """
        Generate random aberration combinations
        """
        # Seed random number generator
        seed(1)

        # Initialise aberration matrix
        aberr_matrix = np.zeros([config['ML']['aberr_num'], config['ML']['zern_num']])

        # Generate random numbers 
        aberr_matrix[:,2] = rand(config['ML']['aberr_num']) * config['ML']['zern_amp']
        aberr_matrix[:,4:] = rand(config['ML']['aberr_num'], config['ML']['zern_num'] - 4) * config['ML']['zern_amp']

        return aberr_matrix

    def rebin(self, arr, new_shape):
        """
        Function to bin a 2D array to the shape specified by new_shape
        """
        shape = (new_shape[0], arr.shape[0] // new_shape[0], new_shape[1], arr.shape[1] // new_shape[1])

        return arr.reshape(shape).mean(-1).mean(1)

    def reflect_process(self):
        """
        Simulates the process of a beam reflecting off a specimen then arriving at the SHWS
        """
        # Get detection path phase by flipping original phase left/right up/down
        self.phase_out = np.flipud(np.fliplr(self.phase_init))

        # Get pupil function and detection path pupil function from phase aberrations and multiply with phase mask
        self.pupil_func = np.exp(-2 * np.pi * 1j / config['AO']['lambda'] * self.phase_init) * self.phase_mask
        self.pupil_func_out = np.exp(-2 * np.pi * 1j / config['AO']['lambda'] * self.phase_out) * self.phase_mask

        # Bin both pupil functions to the size of object space grid
        self.pupil_func_binned = self.rebin(self.pupil_func, (config['ML']['obj_grid_size'], \
            config['ML']['obj_grid_size']))
        self.pupil_func_out_binned = self.rebin(self.pupil_func_out, (config['ML']['obj_grid_size'], \
            config['ML']['obj_grid_size']))
        
        # Pad pupil function with zeros before Fourier transform
        self.pupil_func_pad = np.pad(self.pupil_func_binned, (np.shape(self.pupil_func_binned)[0] // 2, \
            np.shape(self.pupil_func_binned)[0] // 2), 'constant', constant_values = (0, 0))

        # Perform Fourier transform and shift zero frequency components to centre to get amplitude PSF
        self.amp_PSF = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.pupil_func_pad)))

        # Normalise amplitude PSF
        self.amp_PSF = self.amp_PSF / np.amax(self.amp_PSF)

        # Crop amplitude PSF to convolve with sample reflectance profile
        start_1 = (np.shape(self.amp_PSF)[0] - config['ML']['obj_grid_size']) // 2
        self.amp_PSF_crop = self.amp_PSF[start_1 : start_1 + config['ML']['obj_grid_size'], \
            start_1 : start_1 + config['ML']['obj_grid_size']]

        # Generate reflection amplitude PSF
        self.reflect_amp_PSF = self.amp_PSF_crop * self.samp_prof

        # Pad reflection amplitude PSF with zeros before inverse Fourier transform
        self.reflect_amp_PSF_pad = np.pad(self.reflect_amp_PSF, (np.shape(self.reflect_amp_PSF)[0] // 2, \
            np.shape(self.reflect_amp_PSF)[0] // 2), 'constant', constant_values = (0, 0))

        # Perform inverse Fourier transform
        self.pupil_func_2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.reflect_amp_PSF_pad)))

        # Crop reflection pupil function
        start_2 = (np.shape(self.pupil_func_2)[0] - config['ML']['obj_grid_size']) // 2
        self.pupil_func_2 = self.pupil_func_2[start_1 : start_1 + config['ML']['obj_grid_size'], \
            start_1 : start_1 + config['ML']['obj_grid_size']]

        # Multiply reflection pupil function with detection path pupil function to get final pupil function
        self.pupil_func_3 = self.pupil_func_2 * self.pupil_func_out_binned

        # Get wrapped angle of detected pupil function
        self.angle_det_wrapped = np.arctan2(np.imag(self.pupil_func_3), np.real(self.pupil_func_3))

        # Perform angle unwrapping
        self.angle_det_unwrapped = unwrap_phase(self.angle_det_wrapped)
        
        # Get unwrapped phase
        self.phase_det = self.angle_det_unwrapped / (2 * np.pi / config['AO']['lambda'])

        # Interpolate detection phase profile to size of sensor
        mag_fac = self.SB_settings['sensor_width'] / np.shape(self.phase_det)[0]
        self.phase_det = sp.ndimage.zoom(self.phase_det, mag_fac, prefilter = True)
        
        # Apply phase mask to detection phase profile
        self.phase_det = self.phase_det * self.phase_mask

        return self.phase_det

    def acq_centroid(self, image):
        """
        Calculates S-H spot centroids
        """
        # Initialise actual S-H spot centroid coords array and sharpness metric array
        act_cent_coord, act_cent_coord_x, act_cent_coord_y, sharpness = (np.zeros(self.SB_settings['act_ref_cent_num']) for i in range(4))
    
        # Initialise list for storing S-H spot centroid information for all images in data
        slope_x_list, slope_y_list = ([] for i in range(2))

        # Calculate actual S-H spot centroids for each search block
        for m in range(self.SB_settings['act_ref_cent_num']):
            
            # Initialise temporary summing parameters
            sum_x, sum_y, sum_pix = (0 for i in range(3))

            # Get 2D coords of pixels in each search block that need to be summed
            if self.SB_settings['odd_pix']:
                SB_pix_coord_x = np.arange(math.ceil(math.ceil(self.SB_settings['act_ref_cent_coord_x'][m]) - self.SB_settings['SB_rad']), \
                    (math.ceil(math.ceil(self.SB_settings['act_ref_cent_coord_x'][m]) - self.SB_settings['SB_rad']) + self.SB_settings['SB_diam']))
                SB_pix_coord_y = np.arange(math.ceil(math.ceil(self.SB_settings['act_ref_cent_coord_y'][m]) - self.SB_settings['SB_rad']), \
                    (math.ceil(math.ceil(self.SB_settings['act_ref_cent_coord_y'][m]) - self.SB_settings['SB_rad']) + self.SB_settings['SB_diam']))
            else:
                SB_pix_coord_x = np.arange(math.ceil(math.ceil(self.SB_settings['act_ref_cent_coord_x'][m]) - self.SB_settings['SB_rad']) + 1, \
                    (math.ceil(math.ceil(self.SB_settings['act_ref_cent_coord_x'][m]) - self.SB_settings['SB_rad']) + self.SB_settings['SB_diam'] + 1))
                SB_pix_coord_y = np.arange(math.ceil(math.ceil(self.SB_settings['act_ref_cent_coord_y'][m]) - self.SB_settings['SB_rad']) + 1, \
                    (math.ceil(math.ceil(self.SB_settings['act_ref_cent_coord_y'][m]) - self.SB_settings['SB_rad']) + self.SB_settings['SB_diam'] + 1))

            # Crop image within each search area
            image_crop = image[SB_pix_coord_y[0] : SB_pix_coord_y[0] + len(SB_pix_coord_y), \
                SB_pix_coord_x[0] : SB_pix_coord_x[0] + len(SB_pix_coord_x)]

            # Calculate centroid using CoG method
            xx, yy = np.meshgrid(np.arange(SB_pix_coord_x[0], SB_pix_coord_x[0] + len(SB_pix_coord_x)), \
                np.arange(SB_pix_coord_y[0], SB_pix_coord_y[0] + len(SB_pix_coord_y)))

            sum_x = (xx * image_crop).sum()
            sum_y = (yy * image_crop).sum()
            sum_pix = image_crop.sum()

            # Get actual centroid coordinates
            act_cent_coord_x[m] = sum_x / sum_pix
            act_cent_coord_y[m] = sum_y / sum_pix

        # Calculate raw slopes in each dimension
        slope_x = act_cent_coord_x - (self.SB_settings['act_ref_cent_coord_x'].astype(int) + 1)
        slope_y = act_cent_coord_y - (self.SB_settings['act_ref_cent_coord_y'].astype(int) + 1)
        
        # Append slopes to list
        slope_x_list.append(slope_x)
        slope_y_list.append(slope_y)

        return slope_x_list, slope_y_list
    
    @Slot(object)
    def run(self):
        try:
            # Start thread
            self.start.emit()

            self.message.emit('\nProcess started for generating ML dataset...')

            prev1 = time.perf_counter()

            # Generate relevant amounts of tip/tilt for scanning across sample
            tilt_array = -np.linspace(-config['ML']['tilt_amp'], config['ML']['tilt_amp'], config['ML']['scan_num_x'])
            tip_array = -np.linspace(-config['ML']['tip_amp'], config['ML']['tip_amp'], config['ML']['scan_num_y'])

            # Scan each sample with a given amount of aberration combination
            for i in range(config['ML']['samp_num']):

                self.aberr_matrix = self.aberr_gen()

                sp.io.savemat('data/ML_dataset/test/test1/aberr_matrix_sample_' + str(i) + '.mat', dict(aberr_matrix = self.aberr_matrix))

                self.samp_prof = h5py.File('sensorbasedAO/sample_ML/sample' + str(i) + '.mat','r').get('temp_image')

                for j in range(config['ML']['aberr_num']):

                    self.scan_aberr_det = np.zeros([config['ML']['scan_num_x'] * config['ML']['scan_num_y'], config['ML']['zern_num']])

                    self.zern_coeff = self.aberr_matrix[j,:]

                    scan_count = 0

                    for l in range(config['ML']['scan_num_y']):

                        for k in range(config['ML']['scan_num_x']): 

                            if (l * config['ML']['scan_num_x'] + k + 1) == 1 or (l * config['ML']['scan_num_x'] + k + 1) % 50 == 0:
                                print('On sample {}, aberration {}, scan point {}'.format(i + 1, j + 1, l * config['ML']['scan_num_x'] + k + 1))

                            try:
                                
                                # Apply corresponding amounts of tip/tilt to scan over sample
                                self.zern_coeff[0] = tip_array[l]
                                self.zern_coeff[1] = tilt_array[k]

                                # Generate ideal zernike phase profile
                                self.phase_init = zern_phase(self.SB_settings, self.zern_coeff)

                                # Apply sample reflectance process to get detection pupil
                                self.det_phase = self.reflect_process()                              

                                # Get simulated S-H spots and append to list
                                self.AO_image, spot_cent_x, spot_cent_y = fft_spot_from_phase(self.SB_settings, self.det_phase)                               
                                # self.image.emit(self.AO_image)

                                # Calculate centroids of S-H spots
                                self.slope_x, self.slope_y = self.acq_centroid(self.AO_image) 

                                # Concatenate slopes into one slope matrix
                                self.slope = (np.concatenate((self.slope_x, self.slope_y), axis = 1)).T

                                # Get detected zernike coefficients from slope matrix
                                self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], self.slope)

                                # Get partial detected zernike coefficients (except tip/tilt/defocus)
                                self.zern_coeff_detect_part = self.zern_coeff_detect.copy()
                                self.zern_coeff_detect_part[[0, 1, 3], 0] = 0

                                # Save scan point detected zernike coefficients
                                self.scan_aberr_det[l * config['ML']['scan_num_x'] + k,:] = self.zern_coeff_detect[:20, 0].T

                                # Decide whether to output matrix to file
                                if l == (config['ML']['scan_num_y'] - 1) and k == (config['ML']['scan_num_x'] - 1):
                                    sp.io.savemat('data/ML_dataset/test/test1/samp_' + str(i) + '_aberr_' + str(j) + \
                                        '.mat', dict(scan_point_aberrations = self.scan_aberr_det))
                                    prev2 = time.perf_counter()
                                    print('Time lapsed after sample {}, aberration {} is: {} s'.format(i + 1, j + 1, prev2 - prev1))

                                # Increase scan count
                                scan_count += 1

                            except Exception as e:
                                print(e)

            self.message.emit('\nProcess complete.')

            prev3 = time.perf_counter()
            print('Time for entire data generation process is: {} s'.format(prev3 - prev1))

            # Finished ML dataset generation process
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)