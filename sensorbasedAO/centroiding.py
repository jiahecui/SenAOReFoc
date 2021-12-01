from PySide2.QtCore import QObject, Signal, Slot

import h5py
import time
import numpy as np

import log
from config import config
from HDF5_dset import dset_append, get_dset
from image_acquisition import acq_image
from centroid_acquisition import acq_centroid

logger = log.get_logger(__name__)

class Centroiding(QObject):
    """
    Performs system aberration calibration
    """
    start = Signal()
    write = Signal()
    done = Signal()
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

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_0']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_1']

        # Initialise array to store parameters during correction loop
        self.voltages = np.zeros([self.actuator_num, self.AO_settings['loop_max'] + 1])
        self.loop_rms_slopes = np.zeros([self.AO_settings['loop_max'] + 1, 1])
        self.loop_rms_zern, self.loop_rms_zern_part = (np.zeros([self.AO_settings['loop_max'] + 1, 1]) for i in range(2))
        self.strehl = np.zeros([self.AO_settings['loop_max'] + 1, 1])

        super().__init__()

    @Slot(object)
    def run(self):
        try:
            # Set process flags
            self.log = True
            self.loop = True

            # Start thread
            self.start.emit()

            """
            Perform closed-loop system aberration correction or load voltages for DM system flat file on DM 
            """
            # Initialise centroiding information parameter
            self.cent_info = {}

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            # Calibrate system aberration
            if self.debug:

                print('')
                self.message.emit('\nExiting dummy system aberration calibration process.')

            else:

                # Select system aberration calibration mode
                if config['sys_calib']['sys_calib_mode'] == 0:

                    # Load DM system flat file
                    data_file = h5py.File('data_info.h5', 'r+')
                    cent_data = data_file['centroiding_info']
                    voltages = cent_data.get('sys_flat_volts')

                    # Send values vector to mirror
                    self.mirror.Send(voltages)

                    # Wait for DM to settle
                    time.sleep(config['DM']['settling_time'])

                    # Acquire S-H spot image
                    cent_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                    cent_image = np.mean(cent_image_stack, axis = 2)
                    
                    # Image thresholding to remove background
                    cent_image = cent_image - config['image']['threshold'] * np.amax(cent_image)
                    cent_image[cent_image < 0] = 0
                    self.image.emit(cent_image)

                else:

                    # Create new datasets in HDF5 file to store centroiding data
                    get_dset(self.SB_settings, 'centroiding_img', flag = 3)
                    data_file = h5py.File('data_info.h5', 'a')
                    data_set = data_file['centroiding_img']

                    # Run closed-loop control for maximum loop iteration
                    for i in range(self.AO_settings['loop_max'] + 1):

                        if self.loop:

                            try:

                                # Update mirror control voltages
                                if i == 0:

                                    voltages[:] = config['DM']['vol_bias']

                                else:

                                    voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_slopes'], slope_err))

                                # Send values vector to mirror
                                self.mirror.Send(voltages)
                        
                                # Wait for DM to settle
                                time.sleep(config['DM']['settling_time'])

                                # Acquire S-H spot image
                                cent_image_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                                cent_image = np.mean(cent_image_stack, axis = 2)
                                
                                # Image thresholding to remove background
                                cent_image = cent_image - config['image']['threshold'] * np.amax(cent_image)
                                cent_image[cent_image < 0] = 0
                                self.image.emit(cent_image)

                                # Append image to list
                                dset_append(data_set, 'real_cent_img', cent_image)
                            
                                # Calculate centroids of S-H spots
                                act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 0)
                                act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                                # Draw actual S-H spot centroids on image layer
                                cent_image.ravel()[act_cent_coord.astype(int)] = 0
                                self.image.emit(cent_image)

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
                            
                                self.message.emit('\nStrehl ratio {} is: {}.'.format(i, strehl))

                                # Append data to list
                                dset_append(data_set, 'real_spot_slope_x', slope_x)
                                dset_append(data_set, 'real_spot_slope_y', slope_y)
                                dset_append(data_set, 'real_spot_slope', slope)
                                dset_append(data_set, 'real_spot_slope_err', slope_err)
                                dset_append(data_set, 'real_spot_zern_err', zern_err)

                                # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                                if strehl >= config['AO']['tolerance_fact_strehl']:
                                    break  

                            except Exception as e:
                                print(e)
                        else:

                            self.done.emit()

            if not self.debug and config['sys_calib']['sys_calib_mode']:

                # Close HDF5 file
                data_file.close()

            self.message.emit('\nSystem aberration calibration process finished.')

            """
            Returns system aberration information into self.cent_info
            """ 
            if self.log and not self.debug and config['sys_calib']['sys_calib_mode']:

                self.cent_info['residual_phase_err_slopes'] = self.loop_rms_slopes
                self.cent_info['residual_phase_err_zern'] = self.loop_rms_zern
                self.cent_info['residual_phase_err_zern_part'] = self.loop_rms_zern_part
                self.cent_info['strehl_ratio'] = self.strehl
                self.cent_info['sys_flat_volts'] = voltages

                self.info.emit(self.cent_info)
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
        self.log = False
        self.loop = False