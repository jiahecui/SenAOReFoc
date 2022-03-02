from PySide2.QtCore import QObject, Signal, Slot

import time
import h5py
from scipy import io
import numpy as np
import scipy as sp

import log
from config import config
from HDF5_dset import dset_append, get_dset
from image_acquisition import acq_image
from centroid_acquisition import acq_centroid

logger = log.get_logger(__name__)

class Data_Collection(QObject):
    """
    Performs different data collection functions
    """
    # Signal class for starting an event
    start = Signal()

    # Signal class for writing collected AO data into HDF5 file
    write = Signal()

    # Signal class for exiting data collection event
    done = Signal()

    # Signal class for raising an error
    error = Signal(object)

    # Signal class for displaying a SH spot image
    image = Signal(object)

    # Signal class for emitting a message in the message box
    message = Signal(object)

    # Signal class for updating collected AO data
    info = Signal(object)

    def __init__(self, sensor, mirror, settings, debug = False):

        # Get debug status
        self.debug = debug

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
        self.AO_info = {'data_collect': {}}

        # Initialise zernike coefficient array
        self.zern_coeff = np.zeros([config['AO']['control_coeff_num'], 1])

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_0']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_1']
        
        super().__init__()

    @Slot(object)
    def run0(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Run closed-loop AO correction for each generated zernike mode aberration via Zernike control for all 'control_coeff_num' modes 
            and multiple amplitudes of incremental steps 
            """
            # Save calibration slope values and zernike influence function to file
            sp.io.savemat('data/data_collection_0/calib_slope_x.mat', dict(calib_slope_x = self.mirror_settings['calib_slope_x']))
            sp.io.savemat('data/data_collection_0/calib_slope_y.mat', dict(calib_slope_y = self.mirror_settings['calib_slope_y']))
            sp.io.savemat('data/data_collection_0/inf_matrix_zern.mat', dict(inf_matrix_zern = self.mirror_settings['inf_matrix_zern']))

            # Get number of Zernike modes to generate
            zern_num = config['AO']['control_coeff_num'] - 2

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            # Initialise array to store accurate Zernike mode voltages
            self.zern_volts = np.zeros([self.actuator_num, config['AO']['control_coeff_num'] - 2, config['data_collect']['incre_num']])
            self.generated_zern_amp = np.zeros([config['AO']['control_coeff_num'] - 2, config['data_collect']['incre_num']])

            self.message.emit('\nProcess started for data collection mode 0.')

            prev1 = time.perf_counter()

            for n in range(config['data_collect']['run_num']):

                if self.debug:

                    self.message.emit('\nExiting dummy data collection mode 0.')
                    break

                # Run closed-loop control for each zernike mode aberration
                for k in range(config['data_collect']['incre_num']):

                    # Initialise AO information parameter
                    self.AO_info = {'data_collect': {}}

                    # Create new datasets in HDF5 file to store closed-loop AO data and open file
                    get_dset(self.SB_settings, 'data_collect', flag = 0)
                    data_file = h5py.File('data_info.h5', 'a')
                    data_set_1 = data_file['AO_img']['data_collect']

                    # Initialise array to store initial and final detected values of critical parameters
                    self.gen_cor_zern_coeff = np.zeros([zern_num * 2, zern_num])
                    self.gen_cor_rms_zern = np.zeros([zern_num * 2, 1])
                    self.gen_cor_strehl = np.zeros([zern_num * 2, 1])

                    # Determine the amplitude to be generated for each Zernike mode
                    zern_amp_gen = config['data_collect']['incre_amp'] * (k + 1)

                    # Determine initial loop gain for generation of each Zernike mode
                    if zern_amp_gen <= 0.2:
                        loop_gain_gen = 0.2
                    elif zern_amp_gen > 0.2:
                        loop_gain_gen = 0.3

                    for j in range(zern_num):

                        self.message.emit('\nOn amplitude {} Zernike mode {}.'.format(k + 1, j + 3))

                        for i in range(self.AO_settings['loop_max'] + 1):
                            
                            if self.loop:
                                
                                try:

                                    # Update mirror control voltages
                                    if i == 0:

                                        if not self.debug:

                                            # Generate one Zernike mode on DM for correction each time
                                            self.zern_coeff[j + 2, 0] = zern_amp_gen

                                            # Run closed-loop to generate a precise amount of Zernike modes using DM
                                            for m in range(config['data_collect']['loop_max_gen']):

                                                if m == 0:

                                                    voltages[:] = config['DM']['vol_bias']

                                                else:

                                                    # Update control voltages
                                                    voltages -= loop_gain_gen * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                        [:,:config['AO']['control_coeff_num']], (zern_array_det[:config['AO']['control_coeff_num']] - self.zern_coeff)))

                                                # Send values vector to mirror
                                                self.mirror.Send(voltages)

                                                # Wait for DM to settle
                                                time.sleep(config['DM']['settling_time'])
                                            
                                                # Acquire S-H spots using camera
                                                AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                                AO_image = np.mean(AO_image_stack, axis = 2)

                                                # Image thresholding to remove background
                                                AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                                AO_image[AO_image < 0] = 0
                                                self.image.emit(AO_image)

                                                # Append image to list
                                                dset_append(data_set_1, 'real_AO_img', AO_image)

                                                # Calculate centroids of S-H spots
                                                act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2)
                                                act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                                # Draw actual S-H spot centroids on image layer
                                                AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                                self.image.emit(AO_image)

                                                # Take tip\tilt off
                                                slope_x -= np.mean(slope_x)
                                                slope_y -= np.mean(slope_y)

                                                # Concatenate slopes into one slope matrix
                                                slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                                # Get detected zernike coefficients from slope matrix
                                                zern_array_det = np.dot(self.mirror_settings['conv_matrix'], slope)

                                                if abs(zern_array_det[j + 2, 0] - zern_amp_gen) / zern_amp_gen <= 0.075 or m == config['data_collect']['loop_max_gen'] - 1:
                                                    self.message.emit('\nDetected amplitude of mode {} is {} um.'.format(j + 3, zern_array_det[j + 2, 0]))
                                                    self.zern_volts[:, j, k] = voltages
                                                    self.generated_zern_amp[j, k] = zern_array_det[j + 2, 0]
                                                    break
                                        else:

                                            voltages[:] = config['DM']['vol_bias']                              
                                    else:

                                        voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                            [:,:config['AO']['control_coeff_num']], zern_err[:config['AO']['control_coeff_num']]))

                                    # Send values vector to mirror
                                    self.mirror.Send(voltages)
                                    
                                    # Wait for DM to settle
                                    time.sleep(config['DM']['settling_time'])
                                
                                    # Acquire S-H spots using camera
                                    AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                    AO_image = np.mean(AO_image_stack, axis = 2)

                                    # Image thresholding to remove background
                                    AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                    AO_image[AO_image < 0] = 0
                                    self.image.emit(AO_image)

                                    # Append image to list
                                    dset_append(data_set_1, 'real_AO_img', AO_image)

                                    # Calculate centroids of S-H spots
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2) 
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                    # Draw actual S-H spot centroids on image layer
                                    AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                    self.image.emit(AO_image)

                                    # Take tip\tilt off
                                    slope_x -= np.mean(slope_x)
                                    slope_y -= np.mean(slope_y)

                                    # Concatenate slopes into one slope matrix
                                    slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                    # Get detected zernike coefficients from slope matrix
                                    self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                                    # Get residual zernike error and calculate root mean square (rms) error, Strehl ratio
                                    zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                                    zern_err_part[[0, 1], 0] = 0
                                    rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                                    strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)

                                    if i == 0:
                                        self.gen_cor_zern_coeff[2 * j, :] = self.zern_coeff_detect[2:config['AO']['control_coeff_num'], 0].T
                                        self.gen_cor_rms_zern[2 * j, 0] = rms_zern_part
                                        self.gen_cor_strehl[2 * j, 0] = strehl

                                    # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                                    if strehl >= config['AO']['tolerance_fact_strehl'] or i == self.AO_settings['loop_max']:
                                        self.message.emit('\nStrehl ratio {} is {}.'.format(i, strehl))
                                        self.zern_coeff[j + 2] = 0
                                        self.gen_cor_zern_coeff[2 * j + 1, :] = self.zern_coeff_detect[2:config['AO']['control_coeff_num'], 0].T
                                        self.gen_cor_rms_zern[2 * j + 1, 0] = rms_zern_part
                                        self.gen_cor_strehl[2 * j + 1, 0] = strehl
                                        break                 

                                except Exception as e:
                                    print(e)
                            else:

                                self.done.emit()

                    # Save data to file
                    sp.io.savemat('data/data_collection_0/amp_' + str(config['data_collect']['incre_amp'] * (k + 1)) + '_zern_amp_run' + str(n + 1) + '.mat',\
                        dict(zern_amp = self.gen_cor_zern_coeff))
                    sp.io.savemat('data/data_collection_0/amp_' + str(config['data_collect']['incre_amp'] * (k + 1)) + '_rms_zern_run' + str(n + 1) + '.mat',\
                        dict(rms_zern = self.gen_cor_rms_zern))
                    sp.io.savemat('data/data_collection_0/amp_' + str(config['data_collect']['incre_amp'] * (k + 1)) + '_strehl_run' + str(n + 1) + '.mat',\
                        dict(strehl = self.gen_cor_strehl))

                    # Close HDF5 file
                    data_file.close()

            if not self.debug:

                # Save accurate Zernike mode voltages and amplitudes to file
                sp.io.savemat('data/data_collection_0/zern_volts_' + str(config['data_collect']['incre_num']) + '_' + str(config['data_collect']['incre_amp']) + '.mat',\
                    dict(zern_volts = self.zern_volts))
                sp.io.savemat('data/data_collection_0/generated_zern_amp_' + str(config['data_collect']['incre_num']) + '_' + str(config['data_collect']['incre_amp']) + '.mat',\
                    dict(generated_zern_amp = self.generated_zern_amp))

            prev2 = time.perf_counter()
            self.message.emit('\nTime for data collection mode 0 is: {} s.'.format(prev2 - prev1))

            """
            Returns data collection results into self.AO_info
            """             
            if self.log and not self.debug:
                
                self.AO_info['data_collect']['zern_gen_cor_zern_amp_via_zern'] = self.gen_cor_zern_coeff
                self.AO_info['data_collect']['zern_gen_cor_rms_zern_via_zern'] = self.gen_cor_rms_zern
                self.AO_info['data_collect']['zern_gen_cor_strehl_via_zern'] = self.gen_cor_strehl

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit()

            self.done.emit()

        except Exception as e:
            self.error.emit(e)
            raise

    @Slot(object)
    def run1(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Run closed-loop AO correction for each generated zernike mode aberration via slopes control for all 'control_coeff_num' modes 
            and multiple amplitudes of incremental steps 
            """
            # Save calibration slope values and zernike influence function to file
            sp.io.savemat('data/data_collection_1/calib_slope_x.mat', dict(calib_slope_x = self.mirror_settings['calib_slope_x']))
            sp.io.savemat('data/data_collection_1/calib_slope_y.mat', dict(calib_slope_y = self.mirror_settings['calib_slope_y']))
            sp.io.savemat('data/data_collection_1/inf_matrix_zern.mat', dict(inf_matrix_zern = self.mirror_settings['inf_matrix_zern']))

            # Get number of Zernike modes to generate
            zern_num = config['AO']['control_coeff_num'] - 2

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            # Initialise array to store accurate Zernike mode voltages
            self.zern_volts = np.zeros([self.actuator_num, config['AO']['control_coeff_num'] - 2, config['data_collect']['incre_num']])
            self.generated_zern_amp = np.zeros([config['AO']['control_coeff_num'] - 2, config['data_collect']['incre_num']])

            self.message.emit('\nProcess started for data collection mode 1.')

            prev1 = time.perf_counter()

            for n in range(config['data_collect']['run_num']):

                if self.debug:

                    self.message.emit('\nExiting dummy data collection mode 1.')
                    break

                # Run closed-loop control for each zernike mode aberration
                for k in range(config['data_collect']['incre_num']):

                    # Initialise AO information parameter
                    self.AO_info = {'data_collect': {}}

                    # Create new datasets in HDF5 file to store closed-loop AO data and open file
                    get_dset(self.SB_settings, 'data_collect', flag = 0)
                    data_file = h5py.File('data_info.h5', 'a')
                    data_set_1 = data_file['AO_img']['data_collect']

                    # Initialise array to store initial and final detected value of critical parameters
                    self.gen_cor_zern_coeff = np.zeros([zern_num * 2, zern_num])
                    self.gen_cor_rms_zern = np.zeros([zern_num * 2, 1])
                    self.gen_cor_strehl = np.zeros([zern_num * 2, 1])

                    # Determine the amplitude to be generated for each Zernike mode
                    zern_amp_gen = config['data_collect']['incre_amp'] * (k + 1)

                    # Determine initial loop gain for generation of each Zernike mode
                    if zern_amp_gen <= 0.2:
                        loop_gain_gen = 0.2
                    elif zern_amp_gen > 0.2:
                        loop_gain_gen = 0.3

                    for j in range(zern_num):

                        self.message.emit('\nOn amplitude {} Zernike mode {}.'.format(k + 1, j + 3))

                        for i in range(self.AO_settings['loop_max'] + 1):
                            
                            if self.loop:
                                
                                try:

                                    # Update mirror control voltages
                                    if i == 0:

                                        if not self.debug:

                                            # Generate one Zernike mode on DM for correction each time
                                            self.zern_coeff[j + 2, 0] = zern_amp_gen

                                            # Run closed-loop to generate a precise amount of Zernike modes using DM
                                            for m in range(config['data_collect']['loop_max_gen']):

                                                if m == 0:

                                                    voltages[:] = config['DM']['vol_bias']

                                                else:

                                                    # Update control voltages
                                                    voltages -= loop_gain_gen * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                        [:,:config['AO']['control_coeff_num']], (zern_array_det[:config['AO']['control_coeff_num']] - self.zern_coeff)))

                                                # Send values vector to mirror
                                                self.mirror.Send(voltages)

                                                # Wait for DM to settle
                                                time.sleep(config['DM']['settling_time'])
                                            
                                                # Acquire S-H spots using camera
                                                AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                                AO_image = np.mean(AO_image_stack, axis = 2)

                                                # Image thresholding to remove background
                                                AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                                AO_image[AO_image < 0] = 0
                                                self.image.emit(AO_image)

                                                # Append image to list
                                                dset_append(data_set_1, 'real_AO_img', AO_image)

                                                # Calculate centroids of S-H spots
                                                act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2)
                                                act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                                # Draw actual S-H spot centroids on image layer
                                                AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                                self.image.emit(AO_image)

                                                # Take tip\tilt off
                                                slope_x -= np.mean(slope_x)
                                                slope_y -= np.mean(slope_y)

                                                # Concatenate slopes into one slope matrix
                                                slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                                # Get detected zernike coefficients from slope matrix
                                                zern_array_det = np.dot(self.mirror_settings['conv_matrix'], slope)                                                

                                                if abs(zern_array_det[j + 2, 0] - zern_amp_gen) / zern_amp_gen <= 0.075 or m == config['data_collect']['loop_max_gen'] - 1:
                                                    self.message.emit('\nDetected amplitude of mode {} is {} um.'.format(j + 3, zern_array_det[j + 2, 0]))
                                                    self.zern_volts[:, j, k] = voltages
                                                    self.generated_zern_amp[j, k] = zern_array_det[j + 2, 0]
                                                    break
                                        else:

                                            voltages[:] = config['DM']['vol_bias']
                                    else:

                                        voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_slopes'], slope_err))

                                    # Send values vector to mirror
                                    self.mirror.Send(voltages)                                  
                                    
                                    # Wait for DM to settle
                                    time.sleep(config['DM']['settling_time'])
                                
                                    # Acquire S-H spots using camera
                                    AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                    AO_image = np.mean(AO_image_stack, axis = 2)

                                    # Image thresholding to remove background
                                    AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                    AO_image[AO_image < 0] = 0
                                    self.image.emit(AO_image)

                                    # Append image to list
                                    dset_append(data_set_1, 'real_AO_img', AO_image)

                                    # Calculate centroids of S-H spots
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2) 
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                    # Draw actual S-H spot centroids on image layer
                                    AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                    self.image.emit(AO_image)

                                    # Take tip\tilt off
                                    slope_x -= np.mean(slope_x)
                                    slope_y -= np.mean(slope_y)

                                    # Concatenate slopes into one slope matrix
                                    slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                    # Get residual slope error
                                    slope_err = slope.copy()

                                    # Get detected zernike coefficients from slope matrix
                                    self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                                    # Get residual zernike error and calculate root mean square (rms) error, Strehl ratio
                                    zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                                    zern_err_part[[0, 1], 0] = 0
                                    rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                                    strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)
                                    
                                    if i == 0:
                                        self.gen_cor_zern_coeff[2 * j, :] = self.zern_coeff_detect[2:config['AO']['control_coeff_num'], 0].T
                                        self.gen_cor_rms_zern[2 * j, 0] = rms_zern_part
                                        self.gen_cor_strehl[2 * j, 0] = strehl

                                    # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                                    if strehl >= config['AO']['tolerance_fact_strehl'] or i == self.AO_settings['loop_max']:
                                        self.message.emit('\nStrehl ratio {} is {}.'.format(i, strehl))
                                        self.zern_coeff[j + 2] = 0
                                        self.gen_cor_zern_coeff[2 * j + 1, :] = self.zern_coeff_detect[2:config['AO']['control_coeff_num'], 0].T
                                        self.gen_cor_rms_zern[2 * j + 1, 0] = rms_zern_part
                                        self.gen_cor_strehl[2 * j + 1, 0] = strehl
                                        break                 

                                except Exception as e:
                                    print(e)
                            else:

                                self.done.emit()

                    # Save data to file
                    sp.io.savemat('data/data_collection_1/amp_' + str(config['data_collect']['incre_amp'] * (k + 1)) + '_zern_amp_run' + str(n + 1) + '.mat',\
                        dict(zern_amp = self.gen_cor_zern_coeff))
                    sp.io.savemat('data/data_collection_1/amp_' + str(config['data_collect']['incre_amp'] * (k + 1)) + '_rms_zern_run' + str(n + 1) + '.mat',\
                        dict(rms_zern = self.gen_cor_rms_zern))
                    sp.io.savemat('data/data_collection_1/amp_' + str(config['data_collect']['incre_amp'] * (k + 1)) + '_strehl_run' + str(n + 1) + '.mat',\
                        dict(strehl = self.gen_cor_strehl))

                    # Close HDF5 file
                    data_file.close()

            if not self.debug:

                # Save accurate Zernike mode voltages to file
                sp.io.savemat('data/data_collection_1/zern_volts_' + str(config['data_collect']['incre_num']) + '_' + str(config['data_collect']['incre_amp']) + '.mat',\
                    dict(zern_volts = self.zern_volts))
                sp.io.savemat('data/data_collection_1/generated_zern_amp_' + str(config['data_collect']['incre_num']) + '_' + str(config['data_collect']['incre_amp']) + '.mat',\
                    dict(generated_zern_amp = self.generated_zern_amp))

            prev2 = time.perf_counter()
            self.message.emit('\nTime for data collection mode 1 is: {}.'.format(prev2 - prev1))

            """
            Returns data collection results into self.AO_info
            """             
            if self.log and not self.debug:
                
                self.AO_info['data_collect']['zern_gen_cor_zern_amp_via_slopes'] = self.gen_cor_zern_coeff
                self.AO_info['data_collect']['zern_gen_cor_rms_zern_via_slopes'] = self.gen_cor_rms_zern
                self.AO_info['data_collect']['zern_gen_cor_strehl_via_slopes'] = self.gen_cor_strehl

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit()

            self.done.emit()

        except Exception as e:
            self.error.emit(e)
            raise

    @Slot(object)
    def run2(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Run closed-loop AO correction for certain combinations of zernike mode aberrations via Zernike control
            """
            # Save calibration slope values and zernike influence function to file
            sp.io.savemat('data/data_collection_2/calib_slope_x.mat', dict(calib_slope_x = self.mirror_settings['calib_slope_x']))
            sp.io.savemat('data/data_collection_2/calib_slope_y.mat', dict(calib_slope_y = self.mirror_settings['calib_slope_y']))
            sp.io.savemat('data/data_collection_2/inf_matrix_zern.mat', dict(inf_matrix_zern = self.mirror_settings['inf_matrix_zern']))

            # Initialise AO information parameter
            self.AO_info = {'data_collect': {}}

            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'data_collect', flag = 0)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['data_collect']

            # Retrieve input zernike coefficient array
            zern_array_temp = np.array(self.SB_settings['zernike_array_test'])
            zern_array = np.zeros([config['AO']['control_coeff_num'], 1])
            zern_array[:len(zern_array_temp), 0] = zern_array_temp

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            self.message.emit('\nProcess started for data collection mode 2.')

            prev1 = time.perf_counter()

            for n in range(config['data_collect']['run_num']):

                if self.debug:

                    self.message.emit('\nExiting dummy data collection mode 2.')
                    break

                # Initialise array to store detected values of critical parameters
                self.gen_cor_zern_coeff = np.zeros([self.AO_settings['loop_max'] + 1, config['AO']['control_coeff_num'] - 2])
                self.gen_cor_rms_zern = np.zeros([self.AO_settings['loop_max'] + 1, 1])
                self.gen_cor_strehl = np.zeros([self.AO_settings['loop_max'] + 1, 1])

                # Get initial loop_gain_gen
                loop_gain_gen = config['AO']['loop_gain']                  

                for i in range(self.AO_settings['loop_max'] + 1):
                    
                    if self.loop:
                        
                        try:

                            # Update mirror control voltages
                            if i == 0:

                                if not self.debug:

                                    # Generate Zernike modes on DM for correction
                                    self.zern_coeff = zern_array.copy()

                                    # Run closed-loop to generate a precise amount of Zernike modes using DM
                                    for m in range(config['data_collect']['loop_max_gen']):

                                        if m == 0:

                                            voltages[:] = config['DM']['vol_bias']

                                        else:

                                            # Update control voltages
                                            voltages -= loop_gain_gen * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                [:,:config['AO']['control_coeff_num']], (zern_array_det[:config['AO']['control_coeff_num']] - self.zern_coeff)))

                                        # Send values vector to mirror
                                        self.mirror.Send(voltages)

                                        # Wait for DM to settle
                                        time.sleep(config['DM']['settling_time'])
                                    
                                        # Acquire S-H spots using camera
                                        AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                        AO_image = np.mean(AO_image_stack, axis = 2)

                                        # Image thresholding to remove background
                                        AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                        AO_image[AO_image < 0] = 0
                                        self.image.emit(AO_image)

                                        # Append image to list
                                        dset_append(data_set_1, 'real_AO_img', AO_image)

                                        # Calculate centroids of S-H spots
                                        act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2)
                                        act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                        # Draw actual S-H spot centroids on image layer
                                        AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                        self.image.emit(AO_image)

                                        # Take tip\tilt off
                                        slope_x -= np.mean(slope_x)
                                        slope_y -= np.mean(slope_y)

                                        # Concatenate slopes into one slope matrix
                                        slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                        # Get detected zernike coefficients from slope matrix
                                        zern_array_det = np.dot(self.mirror_settings['conv_matrix'], slope)

                                    self.message.emit('\nZernike mode combination generated.')

                                else:

                                    voltages[:] = config['DM']['vol_bias']                              
                            else:

                                voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                    [:,:config['AO']['control_coeff_num']], zern_err[:config['AO']['control_coeff_num']]))

                            # Send values vector to mirror
                            self.mirror.Send(voltages)
                            
                            # Wait for DM to settle
                            time.sleep(config['DM']['settling_time'])
                        
                            # Acquire S-H spots using camera
                            AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                            AO_image = np.mean(AO_image_stack, axis = 2)

                            # Image thresholding to remove background
                            AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                            AO_image[AO_image < 0] = 0
                            self.image.emit(AO_image)

                            # Append image to list
                            dset_append(data_set_1, 'real_AO_img', AO_image)

                            # Calculate centroids of S-H spots
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2) 
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                            # Draw actual S-H spot centroids on image layer
                            AO_image.ravel()[act_cent_coord.astype(int)] = 0
                            self.image.emit(AO_image)

                            # Take tip\tilt off
                            slope_x -= np.mean(slope_x)
                            slope_y -= np.mean(slope_y)

                            if i == 0:
                                sp.io.savemat('data/data_collection_2/slope_x_before_run' + str(n + 1) + '.mat', dict(slope_x_before = slope_x))
                                sp.io.savemat('data/data_collection_2/slope_y_before_run' + str(n + 1) + '.mat', dict(slope_y_before = slope_y))

                            # Concatenate slopes into one slope matrix
                            slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                            # Get detected zernike coefficients from slope matrix
                            self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                            # Get residual zernike error and calculate root mean square (rms) error, Strehl ratio
                            zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                            zern_err_part[[0, 1], 0] = 0
                            rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                            strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)

                            self.gen_cor_zern_coeff[i, :] = self.zern_coeff_detect[2:config['AO']['control_coeff_num'], 0].T
                            self.gen_cor_rms_zern[i, 0] = rms_zern_part
                            self.gen_cor_strehl[i, 0] = strehl

                            # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                            if strehl >= config['AO']['tolerance_fact_strehl'] or i == self.AO_settings['loop_max']:
                                self.message.emit('\nStrehl ratio {} is {}.'.format(i, strehl))
                                self.zern_coeff[:, 0] = 0
                                sp.io.savemat('data/data_collection_2/slope_x_after_run' + str(n + 1) + '.mat', dict(slope_x_after = slope_x))
                                sp.io.savemat('data/data_collection_2/slope_y_after_run' + str(n + 1) + '.mat', dict(slope_y_after = slope_y))
                                break                 

                        except Exception as e:
                            print(e)
                    else:

                        self.done.emit()

                # Save data to file
                sp.io.savemat('data/data_collection_2/zern_amp_run' + str(n + 1) + '.mat', dict(zern_amp = self.gen_cor_zern_coeff))
                sp.io.savemat('data/data_collection_2/rms_zern_run' + str(n + 1) + '.mat', dict(rms_zern = self.gen_cor_rms_zern))
                sp.io.savemat('data/data_collection_2/strehl_run' + str(n + 1) + '.mat', dict(strehl = self.gen_cor_strehl))

            # Close HDF5 file
            data_file.close()

            prev2 = time.perf_counter()
            self.message.emit('\nTime for data collection mode 2 is: {}.'.format(prev2 - prev1))

            """
            Returns data collection results into self.AO_info
            """             
            if self.log and not self.debug:
                
                self.AO_info['data_collect']['zern_gen_cor_zern_amp_via_zern'] = self.gen_cor_zern_coeff
                self.AO_info['data_collect']['zern_gen_cor_rms_zern_via_zern'] = self.gen_cor_rms_zern
                self.AO_info['data_collect']['zern_gen_cor_strehl_via_zern'] = self.gen_cor_strehl

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit()

            # Finished closed-loop AO process
            self.done.emit()

        except Exception as e:
            self.error.emit(e)
            raise

    @Slot(object)
    def run3(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Run closed-loop AO correction for certain combinations of zernike mode aberrations via slopes control
            """
            # Save calibration slope values and zernike influence function to file
            sp.io.savemat('data/data_collection_3/calib_slope_x.mat', dict(calib_slope_x = self.mirror_settings['calib_slope_x']))
            sp.io.savemat('data/data_collection_3/calib_slope_y.mat', dict(calib_slope_y = self.mirror_settings['calib_slope_y']))
            sp.io.savemat('data/data_collection_3/inf_matrix_zern.mat', dict(inf_matrix_zern = self.mirror_settings['inf_matrix_zern']))

            # Initialise AO information parameter
            self.AO_info = {'data_collect': {}}

            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'data_collect', flag = 0)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['data_collect']

            # Retrieve input zernike coefficient array
            zern_array_temp = np.array(self.SB_settings['zernike_array_test'])
            zern_array = np.zeros([config['AO']['control_coeff_num'], 1])
            zern_array[:len(zern_array_temp), 0] = zern_array_temp

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            self.message.emit('\nProcess started for data collection mode 3.')

            prev1 = time.perf_counter()

            for n in range(config['data_collect']['run_num']):

                if self.debug:

                    self.message.emit('\nExiting dummy data collection mode 3.')
                    break

                # Initialise array to store detected values of critical parameters
                self.gen_cor_zern_coeff = np.zeros([self.AO_settings['loop_max'] + 1, config['AO']['control_coeff_num'] - 2])
                self.gen_cor_rms_zern = np.zeros([self.AO_settings['loop_max'] + 1, 1])
                self.gen_cor_strehl = np.zeros([self.AO_settings['loop_max'] + 1, 1])

                # Get initial loop_gain_gen
                loop_gain_gen = config['AO']['loop_gain']

                for i in range(self.AO_settings['loop_max'] + 1):
                    
                    if self.loop:
                        
                        try:

                            # Update mirror control voltages
                            if i == 0:

                                if not self.debug:

                                    # Generate Zernike modes on DM for correction
                                    self.zern_coeff = zern_array.copy()

                                    # Run closed-loop to generate a precise amount of Zernike modes using DM
                                    for m in range(config['data_collect']['loop_max_gen']):

                                        if m == 0:

                                            voltages[:] = config['DM']['vol_bias']

                                        else:

                                            # Update control voltages
                                            voltages -= loop_gain_gen * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                [:,:config['AO']['control_coeff_num']], (zern_array_det[:config['AO']['control_coeff_num']] - self.zern_coeff)))

                                        # Send values vector to mirror
                                        self.mirror.Send(voltages)

                                        # Wait for DM to settle
                                        time.sleep(config['DM']['settling_time'])
                                    
                                        # Acquire S-H spots using camera
                                        AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                        AO_image = np.mean(AO_image_stack, axis = 2)

                                        # Image thresholding to remove background
                                        AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                                        AO_image[AO_image < 0] = 0
                                        self.image.emit(AO_image)

                                        # Append image to list
                                        dset_append(data_set_1, 'real_AO_img', AO_image)

                                        # Calculate centroids of S-H spots
                                        act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2)
                                        act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                        # Draw actual S-H spot centroids on image layer
                                        AO_image.ravel()[act_cent_coord.astype(int)] = 0
                                        self.image.emit(AO_image)

                                        # Take tip\tilt off
                                        slope_x -= np.mean(slope_x)
                                        slope_y -= np.mean(slope_y)

                                        # Concatenate slopes into one slope matrix
                                        slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                                        # Get detected zernike coefficients from slope matrix
                                        zern_array_det = np.dot(self.mirror_settings['conv_matrix'], slope)

                                    self.message.emit('\nZernike mode combination generated.')
                                    
                                else:

                                    voltages[:] = config['DM']['vol_bias']                              
                            else:

                                voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_slopes'], slope_err))

                            # Send values vector to mirror
                            self.mirror.Send(voltages)
                            
                            # Wait for DM to settle
                            time.sleep(config['DM']['settling_time'])
                        
                            # Acquire S-H spots using camera
                            AO_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                            AO_image = np.mean(AO_image_stack, axis = 2)

                            # Image thresholding to remove background
                            AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                            AO_image[AO_image < 0] = 0
                            self.image.emit(AO_image)

                            # Append image to list
                            dset_append(data_set_1, 'real_AO_img', AO_image)

                            # Calculate centroids of S-H spots
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 2) 
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                            # Draw actual S-H spot centroids on image layer
                            AO_image.ravel()[act_cent_coord.astype(int)] = 0
                            self.image.emit(AO_image)

                            # Take tip\tilt off
                            slope_x -= np.mean(slope_x)
                            slope_y -= np.mean(slope_y)

                            if i == 0:
                                sp.io.savemat('data/data_collection_3/slope_x_before_run' + str(n + 1) + '.mat', dict(slope_x_before = slope_x))
                                sp.io.savemat('data/data_collection_3/slope_y_before_run' + str(n + 1) + '.mat', dict(slope_y_before = slope_y))

                            # Concatenate slopes into one slope matrix
                            slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                            # Get residual slope error
                            slope_err = slope.copy()
                            
                            # Get detected zernike coefficients from slope matrix
                            self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                            # Get residual zernike error and calculate root mean square (rms) error, Strehl ratio
                            zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                            zern_err_part[[0, 1], 0] = 0
                            rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                            strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)

                            self.gen_cor_zern_coeff[i, :] = self.zern_coeff_detect[2:config['AO']['control_coeff_num'], 0].T
                            self.gen_cor_rms_zern[i, 0] = rms_zern_part
                            self.gen_cor_strehl[i, 0] = strehl

                            # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                            if strehl >= config['AO']['tolerance_fact_strehl'] or i == self.AO_settings['loop_max']:
                                self.message.emit('\nStrehl ratio {} is {}.'.format(i, strehl))
                                self.zern_coeff[:, 0] = 0
                                sp.io.savemat('data/data_collection_3/slope_x_after_run' + str(n + 1) + '.mat', dict(slope_x_after = slope_x))
                                sp.io.savemat('data/data_collection_3/slope_y_after_run' + str(n + 1) + '.mat', dict(slope_y_after = slope_y))
                                break                 

                        except Exception as e:
                            print(e)
                    else:

                        self.done.emit()

                # Save data to file
                sp.io.savemat('data/data_collection_3/zern_amp_run' + str(n + 1) + '.mat', dict(zern_amp = self.gen_cor_zern_coeff))
                sp.io.savemat('data/data_collection_3/rms_zern_run' + str(n + 1) + '.mat', dict(rms_zern = self.gen_cor_rms_zern))
                sp.io.savemat('data/data_collection_3/strehl_run' + str(n + 1) + '.mat', dict(strehl = self.gen_cor_strehl))

            # Close HDF5 file
            data_file.close()

            prev2 = time.perf_counter()
            self.message.emit('\nTime for data collection mode 3 is: {}.'.format(prev2 - prev1))

            """
            Returns data collection results into self.AO_info
            """             
            if self.log and not self.debug:
                
                self.AO_info['data_collect']['zern_gen_cor_zern_amp_via_slopes'] = self.gen_cor_zern_coeff
                self.AO_info['data_collect']['zern_gen_cor_rms_zern_via_slopes'] = self.gen_cor_rms_zern
                self.AO_info['data_collect']['zern_gen_cor_strehl_via_slopes'] = self.gen_cor_strehl

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit()

            # Finished closed-loop AO process
            self.done.emit()

        except Exception as e:
            self.error.emit(e)
            raise

    @Slot(object)
    def stop(self):
        self.loop = False
        self.log = False