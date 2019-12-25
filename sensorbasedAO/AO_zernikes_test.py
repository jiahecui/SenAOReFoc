from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import logging
import sys
import os
import argparse
import time
import h5py
import PIL.Image
import numpy as np

import log
from config import config
from HDF5_dset import dset_append, get_dset
from image_acquisition import acq_image
from centroid_acquisition import acq_centroid
from spot_sim import SpotSim

logger = log.get_logger(__name__)

class AO_Zernikes_Test(QObject):
    """
    Tests closed-loop AO using calibrated zernike control matrix and user defined zernike coefficients
    """
    start = Signal()
    write = Signal()
    done = Signal()
    error = Signal(object)
    image = Signal(object)
    message = Signal(object)
    info = Signal(object)

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

        # Initialise AO information parameter
        self.AO_info = {'zern_test': {}}

        # Initialise zernike coefficient array
        self.zern_coeff = np.zeros([config['AO']['control_coeff_num'], 1])
        
        super().__init__()

    @Slot(object)
    def run(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Set zernike coefficient array to user specified values and load corresponding control voltages to DM, acquire S-H spot 
            pattern and reconstruct wavefront (get zernike coefficients), then apply phase residual (zernike coefficient difference
            between applied and detected) to DM and iterate the process WITH A FIXED GAIN until residual phase error is below Marechel
            criterion or iteration has reached maximum
            """
            # Load user specified zernike coefficients
            self.zern_coeff.ravel()[:len(self.AO_settings['zernike_array_test'])] = self.AO_settings['zernike_array_test']
   
            # Get pseudo-inverse of slope - zernike conversion matrix to translate zernike coefficients into slopes           
            conv_matrix_inv = np.linalg.pinv(self.mirror_settings['conv_matrix'])

            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'zern_test', flag = 0)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['zern_test']
            data_set_2 = data_file['AO_info']['zern_test']

            # Initialise array to record root mean square error after each iteration
            self.loop_rms = np.zeros(config['AO']['loop_max'])

            self.message.emit('Process started for closed-loop AO via Zernikes...')

            prev1 = time.perf_counter()

            # Run closed-loop control until residual phase error is below a certain value or iteration has reached specified maximum
            for i in range(config['AO']['loop_max']):
                
                if self.loop:
                    
                    try:
                        if config['dummy']:

                            # Translate zernike coefficients into slopes to use as theoretical centroids for simulation of S-H spots
                            if i == 0:
                                self.zern_coeff_orig = self.zern_coeff.copy()
                            else:
                                self.zern_coeff -= config['AO']['loop_gain'] * zern_err

                            spot_cent_slope = np.dot(conv_matrix_inv, self.zern_coeff)
                            spot_cent_slope_x, spot_cent_slope_y = (np.zeros([1, self.SB_settings['act_ref_cent_num']]) for i in range(2))
                            spot_cent_slope_x[0, :] = spot_cent_slope[:self.SB_settings['act_ref_cent_num'], 0]
                            spot_cent_slope_y[0, :] = spot_cent_slope[self.SB_settings['act_ref_cent_num']:, 0]

                            # Get simulated S-H spots and append to list
                            spot_img = SpotSim(self.SB_settings)
                            AO_image, spot_cent_x, spot_cent_y = spot_img.SH_spot_sim\
                                (centred = 1, xc = np.ravel(spot_cent_slope_x), yc = np.ravel(spot_cent_slope_y))
                            dset_append(data_set_1, 'dummy_AO_img', AO_image)
                            dset_append(data_set_1, 'dummy_spot_cent_x', spot_cent_x)
                            dset_append(data_set_1, 'dummy_spot_cent_y', spot_cent_y)
                        else:

                            # Update mirror control voltages
                            if i == 0:
                                voltages = np.dot(self.mirror_settings['control_matrix_zern'], self.zern_coeff)
                            else:
                                voltages -= config['AO']['loop_gain'] * np.dot(self.mirror_settings['control_matrix_zern'], zern_err)                        

                            # Send values vector to mirror
                            self.mirror.Send(voltages)
                            
                            # Wait for DM to settle
                            time.sleep(config['DM']['settling_time'])
                        
                            # Acquire S-H spots using camera and append to list
                            AO_image = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)
                            dset_append(data_set_1, 'real_AO_img', AO_image)

                        # Image thresholding to remove background
                        AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                        AO_image[AO_image < 0] = 0
                        self.image.emit(AO_image)

                        # Calculate centroids of S-H spots
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2) 
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                        # Draw actual S-H spot centroids on image layer
                        AO_image.ravel()[act_cent_coord.astype(int)] = 0
                        self.image.emit(AO_image)

                        # Concatenate slopes into one slope matrix
                        slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                        # Get detected zernike coefficients from slope matrix
                        self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                        # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                        zern_err = self.zern_coeff_detect.copy() - self.zern_coeff_orig
                        rms = np.sqrt((zern_err ** 2).mean(axis = 0))[0]
                        self.loop_rms[i] = rms

                        print('Root mean square error {} is {}'.format(i + 1, rms))                        

                        # Append data to list
                        if config['dummy']:
                            dset_append(data_set_2, 'dummy_spot_slope_x', slope_x)
                            dset_append(data_set_2, 'dummy_spot_slope_y', slope_y)
                            dset_append(data_set_2, 'dummy_spot_slope', slope)
                            dset_append(data_set_2, 'dummy_spot_zern_err', zern_err)
                        else:
                            dset_append(data_set_2, 'real_spot_slope_x', slope_x)
                            dset_append(data_set_2, 'real_spot_slope_y', slope_y)
                            dset_append(data_set_2, 'real_spot_slope', slope)
                            dset_append(data_set_2, 'real_spot_zern_err', zern_err)

                        # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                        if rms <= config['AO']['tolerance_fact_zern']:
                            break                 

                    except Exception as e:
                        print(e)
                else:

                    self.done.emit()

            # Close HDF5 file
            data_file.close()

            self.message.emit('Process complete.')
            print('Final root mean square error of detected wavefront is: {} microns'.format(rms))

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is:', (prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:
                
                self.AO_info['zern_test']['loop_num'] = i + 1
                self.AO_info['zern_test']['residual_phase_err_1'] = self.loop_rms

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit()

            # Finished closed-loop AO process
            self.done.emit()

        except Exception as e:
            raise
            self.error.emit(e)

    @Slot(object)
    def stop(self):
        self.loop = False
        self.log = False