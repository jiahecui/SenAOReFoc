from PySide2.QtCore import QObject, Signal, Slot

import time
import click
import h5py
import numpy as np

import log
from config import config
from HDF5_dset import dset_append, get_dset
from image_acquisition import acq_image
from centroid_acquisition import acq_centroid

logger = log.get_logger(__name__)

class AO_Slopes(QObject):
    """
    Runs closed-loop AO using calibrated slope control matrix
    """
    start = Signal()
    write = Signal()
    done = Signal(object)
    done2 = Signal(object)
    error = Signal(object)
    image = Signal(object)
    message = Signal(object)
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

        # Get voltages for remote focusing
        self.remote_focus_voltages = h5py.File('exec_files/RF_calib_volts_interp_full_01um_1501.mat','r').get('interp_volts')
        self.remote_focus_voltages = np.array(self.remote_focus_voltages).T
        
        # Initialise Zernike coefficient array
        self.zern_coeff = np.zeros(config['AO']['control_coeff_num'])

        # Get remote focusing settings on demand and initialise relevant parameters and arrays for closed-loop correction process
        if self.AO_settings['focus_enable'] == 1:
            self.focus_settings = settings['focusing_info']
            self.correct_num = int(self.focus_settings['step_num'])
        else:
            self.correct_num = 1
        self.loop_rms_slopes = np.zeros([self.AO_settings['loop_max'] + 1, self.correct_num])
        self.loop_rms_zern, self.loop_rms_zern_part = (np.zeros([self.AO_settings['loop_max'] + 1, self.correct_num]) for i in range(2))
        self.strehl, self.strehl_2 = (np.zeros([self.AO_settings['loop_max'] + 1, self.correct_num]) for i in range(2))

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_0']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_1']

        # Initialise array to store voltages during correction loop
        self.voltages = np.zeros([self.actuator_num, self.AO_settings['loop_max'] + 1])

        super().__init__()

    @Slot(object)
    def run1(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Normal closed-loop AO process using a FIXED GAIN, iterated until residual phase error is below value given by Marechel 
            criterion or iteration has reached maximum
            """
            # Initialise AO information parameter
            self.AO_info = {'slope_AO_1': {}}
        
            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'slope_AO_1', flag = 2)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['slope_AO_1']
            data_set_2 = data_file['AO_info']['slope_AO_1']

            self.message.emit('\nProcess started for closed-loop AO via slopes...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            prev1 = time.perf_counter()

            # Run closed-loop control until tolerance value or maximum loop iteration is reached
            for i in range(self.AO_settings['loop_max'] + 1):
                
                if self.loop:

                    try:

                        # Update mirror control voltages
                        if i == 0:

                            # Determine whether to generate Zernike modes using DM
                            if not self.debug and config['AO']['zern_gen'] == 1:

                                # Retrieve input zernike coefficient array
                                zern_array_temp = np.array(self.SB_settings['zernike_array_test'])
                                zern_array = np.zeros([config['AO']['control_coeff_num'], 1])
                                zern_array[:len(zern_array_temp), 0] = zern_array_temp

                                mode_index = np.nonzero(zern_array)[0][0]

                                # Determine initial loop gain for generation of each Zernike mode
                                if zern_array[mode_index, 0] <= 0.2:
                                    loop_gain_gen = 0.2
                                elif zern_array[mode_index, 0] > 0.2:
                                    loop_gain_gen = 0.3

                                # Run closed-loop to generate a precise amount of Zernike modes using DM
                                for j in range(config['AO']['loop_max_gen']):

                                    if j == 0:

                                        voltages[:] = config['DM']['vol_bias']

                                    else:

                                        # Update control voltages
                                        voltages -= loop_gain_gen * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                            [:,:config['AO']['control_coeff_num']], (zern_array_det[:config['AO']['control_coeff_num']] - zern_array)))

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
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 4)
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

                                    print('Detected amplitude of mode {} is {} um'.format(mode_index + 1, zern_array_det[mode_index, 0]))

                                    if abs(zern_array_det[mode_index, 0] - zern_array[mode_index, 0]) / zern_array[mode_index, 0] <= 0.075:
                                        break
                                
                                # Ask user whether to proceed with correction
                                self.message.emit('\nPress [y] to proceed with correction.')
                                c = click.getchar()

                                while True:
                                    if c == 'y':
                                        break
                                    else:
                                        self.message.emit('\nInvalid input. Please try again.')

                                    c = click.getchar()

                            else:

                                voltages[:] = config['DM']['vol_bias']
                        else:

                            voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_slopes'], slope_err))

                            print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages))) 

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
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 4)
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                        # Draw actual S-H spot centroids on image layer
                        AO_image.ravel()[act_cent_coord.astype(int)] = 0
                        self.image.emit(AO_image)

                        # Take tip\tilt off
                        slope_x -= np.mean(slope_x)
                        slope_y -= np.mean(slope_y)

                        # Concatenate slopes into one slope matrix
                        slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                        # Get residual slope error and calculate root mean square (rms) error
                        slope_err = slope.copy()
                        rms_slope = np.sqrt((slope_err ** 2).mean())
                        self.loop_rms_slopes[i] = rms_slope

                        # Get detected zernike coefficients from slope matrix
                        self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                        # Get residual zernike error and calculate root mean square (rms) error
                        zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                        zern_err_part[[0, 1], 0] = 0
                        rms_zern = np.sqrt((zern_err ** 2).sum())
                        rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                        self.loop_rms_zern[i] = rms_zern
                        self.loop_rms_zern_part[i] = rms_zern_part

                        strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)
                        self.strehl[i] = strehl
                    
                        print('Strehl ratio {} is: {}'.format(i, strehl))

                        # Append data to list
                        dset_append(data_set_2, 'real_spot_slope_x', slope_x)
                        dset_append(data_set_2, 'real_spot_slope_y', slope_y)
                        dset_append(data_set_2, 'real_spot_slope', slope)
                        dset_append(data_set_2, 'real_spot_slope_err', slope_err)
                        dset_append(data_set_2, 'real_spot_zern_err', zern_err)

                        # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                        if strehl >= config['AO']['tolerance_fact_strehl']:
                            break                 

                    except Exception as e:
                        print(e)
                else:

                    self.done.emit(1)

            # Close HDF5 file
            data_file.close()

            self.message.emit('\nProcess complete.')
            print('Final root mean square error of detected wavefront is: {} um'.format(rms_zern))

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is: {} s'.format(prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['slope_AO_1']['loop_num'] = i
                self.AO_info['slope_AO_1']['residual_phase_err_slopes'] = self.loop_rms_slopes
                self.AO_info['slope_AO_1']['residual_phase_err_zern'] = self.loop_rms_zern
                self.AO_info['slope_AO_1']['residual_phase_err_zern_part'] = self.loop_rms_zern_part
                self.AO_info['slope_AO_1']['strehl_ratio'] = self.strehl

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit(1)

            # Finished closed-loop AO process
            self.done.emit(1)

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
            Closed-loop AO process to deal with obscured S-H spots using a FIXED GAIN, iterated until residual phase error is below value 
            given by Marechel criterion or iteration has reached maximum
            """
            # Initialise AO information parameter
            self.AO_info = {'slope_AO_2': {}}

            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'slope_AO_2', flag = 2)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['slope_AO_2']
            data_set_2 = data_file['AO_info']['slope_AO_2']

            self.message.emit('\nProcess started for closed-loop AO via slopes with obscured subapertures...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            prev1 = time.perf_counter()

            # Run closed-loop control until tolerance value or maximum loop iteration is reached
            for i in range(self.AO_settings['loop_max'] + 1):
                
                if self.loop:

                    try:

                        # Update mirror control voltages
                        if i == 0:

                            # Determine whether to generate Zernike modes using DM
                            if not self.debug and config['AO']['zern_gen'] == 1:

                                # Retrieve input zernike coefficient array
                                zern_array_temp = np.array(self.SB_settings['zernike_array_test'])
                                zern_array = np.zeros([config['AO']['control_coeff_num'], 1])
                                zern_array[:len(zern_array_temp), 0] = zern_array_temp
                                
                                mode_index = np.nonzero(zern_array)[0][0]

                                # Determine initial loop gain for generation of each Zernike mode
                                if zern_array[mode_index, 0] <= 0.2:
                                    loop_gain_gen = 0.2
                                elif zern_array[mode_index, 0] > 0.2:
                                    loop_gain_gen = 0.3

                                # Run closed-loop to generate a precise amount of Zernike modes using DM
                                for j in range(config['AO']['loop_max_gen']):

                                    if j == 0:

                                        voltages[:] = config['DM']['vol_bias']

                                    else:
                                       
                                        # Update control voltages
                                        voltages -= loop_gain_gen * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                            [:,:config['AO']['control_coeff_num']], (zern_array_det[:config['AO']['control_coeff_num']] - zern_array)))

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
                                    act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 6)
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

                                    print('Detected amplitude of mode {} is {} um'.format(mode_index + 1, zern_array_det[mode_index, 0]))

                                    if abs(zern_array_det[mode_index, 0] - zern_array[mode_index, 0]) / zern_array[mode_index, 0] <= 0.75:
                                        break
                                
                                # Ask user whether to proceed with correction
                                self.message.emit('\nPress [y] to proceed with correction.')
                                c = click.getchar()

                                while True:
                                    if c == 'y':
                                        break
                                    else:
                                        self.message.emit('\nInvalid input. Please try again.')

                                    c = click.getchar()

                            else:

                                voltages[:] = config['DM']['vol_bias']
                        else:

                            voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(control_matrix_slopes, slope_err))

                            print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

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

                        # Detect obscured subapertures while calculating centroids of S-H spots
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 6)
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                        # Remove corresponding elements from slopes and rows from influence function matrix
                        if self.debug == 1:
                            index_remove = np.where(slope_x + self.SB_settings['act_ref_cent_coord_x'].astype(int) + 1 == 0)[1]
                        else:
                            index_remove = np.where(slope_x + self.SB_settings['act_ref_cent_coord_x'] == 0)[1]

                        print('Number of obscured subapertures:', np.size(index_remove))
                        print('Index of obscured subapertures:', index_remove)

                        index_remove_inf = np.concatenate((index_remove, index_remove + self.SB_settings['act_ref_cent_num']), axis = None)
                        slope_x = np.delete(slope_x, index_remove, axis = 1)
                        slope_y = np.delete(slope_y, index_remove, axis = 1)
                        act_cent_coord = np.delete(act_cent_coord, index_remove, axis = None)
                        inf_matrix_slopes = np.delete(self.mirror_settings['inf_matrix_slopes'].copy(), index_remove_inf, axis = 0)
                        conv_matrix = np.delete(self.mirror_settings['conv_matrix'].copy(), index_remove_inf, axis = 1)

                        # Draw actual S-H spot centroids on image layer
                        AO_image.ravel()[act_cent_coord.astype(int)] = 0
                        self.image.emit(AO_image)

                        # Calculate singular value decomposition of modified influence function matrix
                        u, s, vh = np.linalg.svd(inf_matrix_slopes, full_matrices = False)
                        # print('u: {}, s: {}, vh: {}'.format(u, s, vh))
                        # print('The shapes of u, s, and vh are: {}, {}, and {}'.format(np.shape(u), np.shape(s), np.shape(vh)))

                        # Recalculate pseudo-inverse of modified influence function matrix to get new control matrix
                        control_matrix_slopes = np.linalg.pinv(inf_matrix_slopes)

                        # Take tip\tilt off
                        slope_x -= np.mean(slope_x)
                        slope_y -= np.mean(slope_y)

                        # Concatenate slopes into one slope matrix
                        slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                        # Get residual slope error and calculate root mean square (rms) error
                        slope_err = slope.copy()
                        rms_slope = np.sqrt((slope_err ** 2).mean())
                        self.loop_rms_slopes[i] = rms_slope

                        # Get detected zernike coefficients from slope matrix
                        self.zern_coeff_detect = np.dot(conv_matrix, slope)

                        # Get residual zernike error and calculate root mean square (rms) error
                        zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                        zern_err_part[[0, 1], 0] = 0
                        rms_zern = np.sqrt((zern_err ** 2).sum())
                        rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                        self.loop_rms_zern[i] = rms_zern
                        self.loop_rms_zern_part[i] = rms_zern_part

                        strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)
                        self.strehl[i] = strehl

                        print('Strehl ratio {} is: {}'.format(i, strehl))                      

                        # Append data to list
                        dset_append(data_set_2, 'real_spot_zern_err', zern_err)

                        # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                        if strehl >= config['AO']['tolerance_fact_strehl']:
                            break                 

                    except Exception as e:
                        print(e)
                else:

                    self.done.emit(2)

            # Close HDF5 file
            data_file.close()

            self.message.emit('\nProcess complete.')
            print('Final root mean square error of detected wavefront is: {} um'.format(rms_zern))

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is: {} s'.format(prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['slope_AO_2']['loop_num'] = i
                self.AO_info['slope_AO_2']['residual_phase_err_slopes'] = self.loop_rms_slopes
                self.AO_info['slope_AO_2']['residual_phase_err_zern'] = self.loop_rms_zern
                self.AO_info['slope_AO_2']['residual_phase_err_zern_part'] = self.loop_rms_zern_part
                self.AO_info['slope_AO_2']['strehl_ratio'] = self.strehl

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit(2)

            # Finished closed-loop AO process
            self.done.emit(2)

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
            Closed-loop AO process to handle partial correction using a FIXED GAIN, iterated until residual phase error is below value given by Marechel 
            criterion or iteration has reached maximum
            """
            # Initialise AO information parameter
            self.AO_info = {'slope_AO_3': {}}

            # Calculate modified influence function with partial correction (suppressing defocus)
            inf_matrix_slopes = np.concatenate((self.mirror_settings['inf_matrix_slopes'], config['AO']['suppress_gain'] * \
                self.mirror_settings['inf_matrix_zern'][[3], :]), axis = 0)

            # Calculate singular value decomposition of modified influence function matrix
            u, s, vh = np.linalg.svd(inf_matrix_slopes, full_matrices = False)
            # print('u: {}, s: {}, vh: {}'.format(u, s, vh))
            # print('The shapes of u, s, and vh are: {}, {}, and {}'.format(np.shape(u), np.shape(s), np.shape(vh)))

            # Calculate pseudo-inverse of modified influence function matrix to get new control matrix
            control_matrix_slopes = np.linalg.pinv(inf_matrix_slopes)
        
            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'slope_AO_3', flag = 2)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['slope_AO_3']
            data_set_2 = data_file['AO_info']['slope_AO_3']

            if self.AO_settings['focus_enable'] == 0:
                self.message.emit('\nProcess started for closed-loop AO via slopes with partial correction...')
            else:
                self.message.emit('\nProcess started for remote focusing + closed-loop AO via slopes with partial correction...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            prev1 = time.perf_counter()

            # Run correction for each focus depth
            for j in range(self.correct_num):

                # Retrieve voltages for remote focusing component
                if self.AO_settings['focus_enable'] == 1:
                    if self.focus_settings['focus_mode_flag'] == 0:
                        RF_index = int(self.focus_settings['focus_depth_defoc'] // config['RF']['step_incre']) + config['RF']['index_offset']
                        voltages_defoc = np.ravel(self.remote_focus_voltages[:, RF_index])
                    else:
                        RF_index = int(self.focus_settings['start_depth_defoc'] // config['RF']['step_incre'] \
                            + self.focus_settings['step_incre_defoc'] // config['RF']['step_incre'] * j) + config['RF']['index_offset']
                        voltages_defoc = np.ravel(self.remote_focus_voltages[:, RF_index])
                else:
                    voltages_defoc = 0

                # Run closed-loop control until tolerance value or maximum loop iteration is reached
                for i in range(self.AO_settings['loop_max'] + 1):
                    
                    if self.loop:

                        try:

                            # Update mirror control voltages
                            if i == 0:

                                # Determine whether to generate Zernike modes using DM
                                if not self.debug and config['AO']['zern_gen'] == 1:

                                    # Retrieve input zernike coefficient array
                                    zern_array_temp = np.array(self.SB_settings['zernike_array_test'])
                                    zern_array = np.zeros([config['AO']['control_coeff_num'], 1])
                                    zern_array[:len(zern_array_temp), 0] = zern_array_temp
                                    
                                    mode_index = np.nonzero(zern_array)[0][0]

                                    # Determine initial loop gain for generation of each Zernike mode
                                    if zern_array[mode_index, 0] <= 0.2:
                                        loop_gain_gen = 0.2
                                    elif zern_array[mode_index, 0] > 0.2:
                                        loop_gain_gen = 0.3

                                    # Run closed-loop to generate a precise amount of Zernike modes using DM
                                    for j in range(config['AO']['loop_max_gen']):

                                        if j == 0:

                                            voltages[:] = config['DM']['vol_bias']

                                        else:

                                            # Update control voltages
                                            voltages -= loop_gain_gen * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                [:,:config['AO']['control_coeff_num']], (zern_array_det[:config['AO']['control_coeff_num']] - zern_array)))

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
                                        act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 8)
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

                                        print('Detected amplitude of mode {} is {} um'.format(mode_index + 1, zern_array_det[mode_index, 0]))

                                        if abs(zern_array_det[mode_index, 0] - zern_array[mode_index, 0]) / zern_array[mode_index, 0] <= 75:
                                            break
                                    
                                    # Ask user whether to proceed with correction
                                    self.message.emit('\nPress [y] to proceed with correction.')
                                    c = click.getchar()

                                    while True:
                                        if c == 'y':
                                            break
                                        else:
                                            self.message.emit('\nInvalid input. Please try again.')

                                        c = click.getchar()

                                else:

                                    voltages[:] = config['DM']['vol_bias'] + voltages_defoc
                            else:

                                voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(control_matrix_slopes, slope_err))

                                print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

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
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 8)
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                            # Draw actual S-H spot centroids on image layer
                            AO_image.ravel()[act_cent_coord.astype(int)] = 0
                            self.image.emit(AO_image)

                            # Take tip\tilt off
                            slope_x -= np.mean(slope_x)
                            slope_y -= np.mean(slope_y)

                            # Concatenate slopes into one slope matrix
                            slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                            # Get residual slope error and calculate root mean square (rms) error
                            slope_err = slope.copy()
                            rms_slope = np.sqrt((slope_err ** 2).mean())
                            self.loop_rms_slopes[i,j] = rms_slope

                            # Get detected zernike coefficients from slope matrix
                            self.zern_coeff_detect = np.dot(self.mirror_settings['conv_matrix'], slope)

                            # Get residual zernike error and calculate root mean square (rms) error
                            zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                            zern_err_part[[0, 1, 3], 0] = 0
                            rms_zern = np.sqrt((zern_err ** 2).sum())
                            rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                            self.loop_rms_zern[i,j] = rms_zern
                            self.loop_rms_zern_part[i,j] = rms_zern_part

                            strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)
                            self.strehl[i,j] = strehl

                            print('Strehl ratio {} is: {}'.format(i, strehl))                            

                            # Append data to list
                            dset_append(data_set_2, 'real_spot_slope_x', slope_x)
                            dset_append(data_set_2, 'real_spot_slope_y', slope_y)
                            dset_append(data_set_2, 'real_spot_slope', slope)
                            dset_append(data_set_2, 'real_spot_slope_err', slope_err)
                            dset_append(data_set_2, 'real_spot_zern_err', zern_err)

                            # Append zeros to end of slope error array to ensure dimension is consistent with new control matrix
                            slope_err = np.append(slope_err, np.zeros([1, 1]), axis = 0)

                            # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                            if strehl >= config['AO']['tolerance_fact_strehl']:
                                break                 

                        except Exception as e:
                            print(e)
                    else:

                        if self.AO_settings['focus_enable'] == 0:
                            self.done.emit(3)
                        elif self.focus_settings['focus_mode_flag'] == 0:
                            self.done2.emit(0)
                        elif self.focus_settings['focus_mode_flag'] == 1:
                            self.done2.emit(1)

                print('Final root mean square error of detected wavefront is: {} um'.format(rms_zern))

            # Close HDF5 file
            data_file.close()

            self.message.emit('\nProcess complete.')

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is: {} s'.format(prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['slope_AO_3']['loop_num'] = i
                self.AO_info['slope_AO_3']['residual_phase_err_slopes'] = self.loop_rms_slopes
                self.AO_info['slope_AO_3']['residual_phase_err_zern'] = self.loop_rms_zern
                self.AO_info['slope_AO_3']['residual_phase_err_zern_part'] = self.loop_rms_zern_part
                self.AO_info['slope_AO_3']['strehl_ratio'] = self.strehl

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                if self.AO_settings['focus_enable'] == 0:
                    self.done.emit(3)
                elif self.focus_settings['focus_mode_flag'] == 0:
                    self.done2.emit(0)
                elif self.focus_settings['focus_mode_flag'] == 1:
                    self.done2.emit(1)

            # Finished closed-loop AO process
            if self.AO_settings['focus_enable'] == 0:
                self.done.emit(3)
            elif self.focus_settings['focus_mode_flag'] == 0:
                self.done2.emit(0)
            elif self.focus_settings['focus_mode_flag'] == 1:
                self.done2.emit(1)

        except Exception as e:
            self.error.emit(e) 
            raise

    @Slot(object)
    def run4(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Closed-loop AO process to handle both obscured S-H spots and partial correction using a FIXED GAIN, iterated until residual 
            phase error is below value given by Marechel criterion or iteration has reached maximum
            """
            # Initialise AO information parameter
            self.AO_info = {'slope_AO_full': {}}

            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'slope_AO_full', flag = 2)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['slope_AO_full']
            data_set_2 = data_file['AO_info']['slope_AO_full']

            if self.AO_settings['focus_enable'] == 0:
                self.message.emit('\nProcess started for full closed-loop AO via slopes...')
            else:
                self.message.emit('\nProcess started for remote focusing + full closed-loop AO via slopes...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            prev1 = time.perf_counter()

            # Run correction for each focus depth
            for j in range(self.correct_num):

                # Retrieve voltages for remote focusing component
                if self.AO_settings['focus_enable'] == 1:
                    if self.focus_settings['focus_mode_flag'] == 0:
                        RF_index = int(self.focus_settings['focus_depth_defoc'] // config['RF']['step_incre']) + config['RF']['index_offset']
                        voltages_defoc = np.ravel(self.remote_focus_voltages[:, RF_index])
                    else:
                        RF_index = int(self.focus_settings['start_depth_defoc'] // config['RF']['step_incre'] \
                            + self.focus_settings['step_incre_defoc'] // config['RF']['step_incre'] * j) + config['RF']['index_offset']
                        voltages_defoc = np.ravel(self.remote_focus_voltages[:, RF_index])
                else:
                    voltages_defoc = 0

                # Run closed-loop control until tolerance value or maximum loop iteration is reached
                for i in range(self.AO_settings['loop_max'] + 1):
                    
                    if self.loop:

                        try:

                            # Update mirror control voltages
                            if i == 0:

                                # Determine whether to generate Zernike modes using DM
                                if not self.debug and config['AO']['zern_gen'] == 1:

                                    # Retrieve input zernike coefficient array
                                    zern_array_temp = np.array(self.SB_settings['zernike_array_test'])
                                    zern_array = np.zeros([config['AO']['control_coeff_num'], 1])
                                    zern_array[:len(zern_array_temp), 0] = zern_array_temp
                                                                        
                                    mode_index = np.nonzero(zern_array)[0][0]

                                    # Determine initial loop gain for generation of each Zernike mode
                                    if zern_array[mode_index, 0] <= 0.2:
                                        loop_gain_gen = 0.2
                                    elif zern_array[mode_index, 0] > 0.2:
                                        loop_gain_gen = 0.3

                                    # Run closed-loop to generate a precise amount of Zernike modes using DM
                                    for j in range(config['AO']['loop_max_gen']):

                                        if j == 0:

                                            voltages[:] = config['DM']['vol_bias']

                                        else:

                                            # Update control voltages
                                            voltages -= loop_gain_gen * np.ravel(np.dot(self.mirror_settings['control_matrix_zern']\
                                                [:,:config['AO']['control_coeff_num']], (zern_array_det[:config['AO']['control_coeff_num']] - zern_array)))

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
                                        act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 10)
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

                                        print('Detected amplitude of mode {} is {} um'.format(mode_index + 1, zern_array_det[mode_index, 0]))

                                        if abs(zern_array_det[mode_index, 0] - zern_array[mode_index, 0]) / zern_array[mode_index, 0] <= 75:
                                            break
                                    
                                    # Ask user whether to proceed with correction
                                    self.message.emit('\nPress [y] to proceed with correction.')
                                    c = click.getchar()

                                    while True:
                                        if c == 'y':
                                            break
                                        else:
                                            self.message.emit('\nInvalid input. Please try again.')

                                        c = click.getchar()

                                else:

                                    voltages[:] = config['DM']['vol_bias'] + voltages_defoc
                            else:
                                
                                voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(control_matrix_slopes, slope_err))

                                print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

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
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 10)
                            act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                            # Remove corresponding elements from slopes and rows from influence function matrix
                            if self.debug == 1:
                                index_remove = np.where(slope_x + self.SB_settings['act_ref_cent_coord_x'].astype(int) + 1 == 0)[1]
                            else:
                                index_remove = np.where(slope_x + self.SB_settings['act_ref_cent_coord_x'] == 0)[1]

                            print('Number of obscured subapertures:', np.shape(index_remove))
                            print('Index of obscured subapertures:', index_remove)

                            index_remove_inf = np.concatenate((index_remove, index_remove + self.SB_settings['act_ref_cent_num']), axis = None)
                            slope_x = np.delete(slope_x, index_remove, axis = 1)
                            slope_y = np.delete(slope_y, index_remove, axis = 1)
                            act_cent_coord = np.delete(act_cent_coord, index_remove, axis = None)
                            inf_matrix_slopes = np.delete(self.mirror_settings['inf_matrix_slopes'].copy(), index_remove_inf, axis = 0)
                            conv_matrix = np.delete(self.mirror_settings['conv_matrix'].copy(), index_remove_inf, axis = 1)

                            # Draw actual S-H spot centroids on image layer
                            AO_image.ravel()[act_cent_coord.astype(int)] = 0
                            self.image.emit(AO_image)

                            # Calculate modified influence function with partial correction (suppressing tip, tilt, and defocus)
                            inf_matrix_slopes = np.concatenate((inf_matrix_slopes, config['AO']['suppress_gain'] * \
                                self.mirror_settings['inf_matrix_zern'][[0, 1, 3], :]), axis = 0)

                            # Calculate singular value decomposition of modified influence function matrix
                            u, s, vh = np.linalg.svd(inf_matrix_slopes, full_matrices = False)
                            # print('u: {}, s: {}, vh: {}'.format(u, s, vh))
                            # print('The shapes of u, s, and vh are: {}, {}, and {}'.format(np.shape(u), np.shape(s), np.shape(vh)))

                            # Recalculate pseudo-inverse of modified influence function matrix to get new control matrix
                            control_matrix_slopes = np.linalg.pinv(inf_matrix_slopes)

                            # Take tip\tilt off
                            slope_x -= np.mean(slope_x)
                            slope_y -= np.mean(slope_y)

                            # Concatenate slopes into one slope matrix
                            slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                            # Get residual slope error and calculate root mean square (rms) error
                            slope_err = slope.copy()
                            rms_slope = np.sqrt((slope_err ** 2).mean())
                            self.loop_rms_slopes[i,j] = rms_slope

                            # Get detected zernike coefficients from slope matrix
                            self.zern_coeff_detect = np.dot(conv_matrix, slope)

                            # Get residual zernike error and calculate root mean square (rms) error
                            zern_err, zern_err_part = (self.zern_coeff_detect.copy() for c in range(2))
                            zern_err_part[[0, 1, 3], 0] = 0
                            rms_zern = np.sqrt((zern_err ** 2).sum())
                            rms_zern_part = np.sqrt((zern_err_part ** 2).sum())
                            self.loop_rms_zern[i,j] = rms_zern
                            self.loop_rms_zern_part[i,j] = rms_zern_part

                            strehl = np.exp(-(2 * np.pi / config['AO']['lambda'] * rms_zern_part) ** 2)
                            self.strehl[i,j] = strehl
                      
                            print('Strehl ratio {} is: {}'.format(i, strehl))                      
                            
                            # Append data to list
                            dset_append(data_set_2, 'real_spot_zern_err', zern_err)

                            # Append zeros to end of slope error array to ensure dimension is consistent with new control matrix
                            slope_err = np.append(slope_err, np.zeros([3, 1]), axis = 0)

                            # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                            if strehl >= config['AO']['tolerance_fact_strehl']:
                                break                 

                        except Exception as e:
                            print(e)
                    else:

                        if self.AO_settings['focus_enable'] == 0:
                            self.done.emit(4)
                        elif self.focus_settings['focus_mode_flag'] == 0:
                            self.done2.emit(0)
                        elif self.focus_settings['focus_mode_flag'] == 1:
                            self.done2.emit(1)

                print('Final root mean square error of detected wavefront is: {} um'.format(rms_zern))

            # Close HDF5 file
            data_file.close()

            self.message.emit('\nProcess complete.')

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is: {} s'.format(prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['slope_AO_full']['loop_num'] = i
                self.AO_info['slope_AO_full']['residual_phase_err_slopes'] = self.loop_rms_slopes
                self.AO_info['slope_AO_full']['residual_phase_err_zern'] = self.loop_rms_zern
                self.AO_info['slope_AO_full']['residual_phase_err_zern_part'] = self.loop_rms_zern_part
                self.AO_info['slope_AO_full']['strehl_ratio'] = self.strehl
                if self.debug:
                    self.AO_info['slope_AO_full']['strehl_ratio_2'] = self.strehl_2

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                if self.AO_settings['focus_enable'] == 0:
                    self.done.emit(4)
                elif self.focus_settings['focus_mode_flag'] == 0:
                    self.done2.emit(0)
                elif self.focus_settings['focus_mode_flag'] == 1:
                    self.done2.emit(1)

            # Finished closed-loop AO process
            if self.AO_settings['focus_enable'] == 0:
                self.done.emit(4)
            elif self.focus_settings['focus_mode_flag'] == 0:
                self.done2.emit(0)
            elif self.focus_settings['focus_mode_flag'] == 1:
                self.done2.emit(1)

        except Exception as e:
            self.error.emit(e)   
            raise

    @Slot(object)
    def stop(self):
        self.loop = False
        self.log = False