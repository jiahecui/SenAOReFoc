from PySide2.QtCore import QObject, Signal, Slot

import time
import h5py
import numpy as np

import log
from config import config
from HDF5_dset import dset_append, get_dset
from image_acquisition import acq_image
from centroid_acquisition import acq_centroid

logger = log.get_logger(__name__)

class Calibration(QObject):
    """
    Calibrates deformable mirror and retrieves influence function + control matrix
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
        self.SB_settings = settings

        # Get sensor instance
        self.sensor = sensor

        # Get mirror instance
        self.mirror = mirror

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
            self.pitch = config['DM0']['pitch']
            self.aperture = config['DM0']['aperture']
            self.pupil_diam = config['search_block']['pupil_diam_0']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']
            self.pitch = config['DM1']['pitch']
            self.aperture = config['DM1']['aperture']
            self.pupil_diam = config['search_block']['pupil_diam_1']

        # Initialise deformable mirror information parameter
        self.mirror_info = {}

        # Initialise influence function matrix
        self.inf_matrix_slopes = np.zeros([2 * self.SB_settings['act_ref_cent_num'], self.actuator_num])
        
        super().__init__()

    def act_coord_1(self, act_diam):
        """
        Calculates actuator position coordinates according to DM geometry (Alpao69)
        """
        xc, yc = (np.zeros(self.actuator_num) for i in range(2))

        for i in range(5):

            xc[i] = -4 * act_diam
            xc[self.actuator_num - 1 - i] = 4 * act_diam
            yc[i] = (2 - i) * act_diam
            yc[self.actuator_num - 1 - i] = (-2 + i) * act_diam

        for i in range(7):

            xc[5 + i] = -3 * act_diam
            xc[self.actuator_num - 6 - i] = 3 * act_diam
            yc[5 + i] = (3 - i) * act_diam
            yc[self.actuator_num - 6 - i] = (-3 + i) * act_diam

        for i in range(9):

            xc[12 + i] = -2 * act_diam
            xc[21 + i] = -act_diam
            xc[30 + i] = 0
            xc[self.actuator_num - 13 - i] = 2 * act_diam
            xc[self.actuator_num - 22 - i] = act_diam
            yc[12 + i] = (4 - i) * act_diam
            yc[21 + i] = (4 - i) * act_diam
            yc[30 + i] = (4 - i) * act_diam
            yc[self.actuator_num - 13 - i] = (-4 + i) * act_diam
            yc[self.actuator_num - 22 - i] = (-4 + i) * act_diam

        return xc, yc

    def act_coord_2(self, act_diam):
        """
        Calculates actuator position coordinates according to DM geometry (Boston140)
        """
        xc, yc = (np.zeros(self.actuator_num) for i in range(2))

        for i in range(10):

            xc[i] = (-5 - 0.5) * act_diam
            xc[self.actuator_num - 1 - i] = (5 + 0.5) * act_diam
            yc[i] = (4 + 0.5 - i) * act_diam
            yc[self.actuator_num - 1 - i] = (-4 - 0.5 + i) * act_diam

        for i in range(120):

            if i in [11, 23, 35, 47, 59] or i > 59:
                xc[10 + i] = (int(((10 + i) - self.actuator_num // 2) // 12) + 0.5) * act_diam
            else:
                xc[10 + i] = (int(((10 + i) - (self.actuator_num // 2 - 1)) // 12) + 0.5) * act_diam

            yc[10 + i] = (-(i % 12) + (5 + 0.5)) * act_diam

        return xc, yc

    @Slot(object)
    def run(self):
        try:
            # Set process flags
            self.calibrate = True
            self.calc_cent = True
            self.calc_inf = True
            self.log = True

            # Start thread
            self.start.emit()

            if self.debug:

                """
                Get real DM calibration files
                """                
                # Get diameter spacing of one actuator
                act_diam = self.pupil_diam / self.aperture * self.pitch / self.SB_settings['pixel_size']

                # Get actuator coordinates
                if config['DM']['DM_num'] == 0:
                    xc, yc = self.act_coord_1(act_diam)
                elif config['DM']['DM_num'] == 1:
                    xc, yc = self.act_coord_2(act_diam)

                s = np.array(h5py.File('exec_files/real_calib_func/inf_matrix_slopes_SV.mat','r').get('inf_matrix_slopes_SV')).T
                self.inf_matrix_slopes = np.array(h5py.File('exec_files/real_calib_func/inf_matrix_slopes.mat','r').get('inf_matrix_slopes')).T
                self.control_matrix_slopes = np.array(h5py.File('exec_files/real_calib_func/control_matrix_slopes.mat','r').get('control_matrix_slopes')).T
                self.slope_x = np.array(h5py.File('exec_files/real_calib_func/calib_slope_x.mat','r').get('calib_slope_x')).T
                self.slope_y = np.array(h5py.File('exec_files/real_calib_func/calib_slope_y.mat','r').get('calib_slope_y')).T
                svd_check_slopes = np.array(h5py.File('exec_files/real_calib_func/svd_check_slopes.mat','r').get('svd_check_slopes')).T

                self.message.emit('\nDM calibration files loaded.')
                    
            else:

                """
                Apply highest and lowest voltage to each actuator individually and retrieve raw slopes of each S-H spot
                """
                # Initialise deformable mirror voltage array
                voltages = np.zeros(self.actuator_num)
                
                prev1 = time.perf_counter()

                # Create new datasets in HDF5 file to store calibration data
                get_dset(self.SB_settings, 'calibration_img', flag = 4)
                data_file = h5py.File('data_info.h5', 'a')
                data_set = data_file['calibration_img']

                self.message.emit('\nDM calibration process started.')
                
                # Poke each actuator first in to vol_max, then to vol_min
                for i in range(self.actuator_num):

                    if self.calibrate:                    

                        try:

                            if (i + 1) % 10 == 0:

                                self.message.emit('On actuator {}'.format(i + 1))

                            # Apply highest voltage
                            voltages[i] = config['DM']['vol_max']
                        
                            # Send values vector to mirror
                            self.mirror.Send(voltages)
                            
                            # Wait for DM to settle
                            time.sleep(config['DM']['settling_time'])
                            
                            # Acquire S-H spot image
                            image_max_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                            image_max = np.mean(image_max_stack, axis = 2)

                            # Image thresholding to remove background
                            image_max = image_max - config['image']['threshold'] * np.amax(image_max)
                            image_max[image_max < 0] = 0
                            self.image.emit(image_max)

                            # Append image to list
                            dset_append(data_set, 'real_calib_img', image_max)

                            # Apply lowest voltage
                            voltages[i] = config['DM']['vol_min']

                            # Send values vector to mirror
                            self.mirror.Send(voltages)

                            # Wait for DM to settle
                            time.sleep(config['DM']['settling_time'])

                            # Acquire S-H spot image
                            image_min_stack = acq_image(self.sensor, self.SB_settings['sensor_height'], self.SB_settings['sensor_width'], acq_mode = 1)
                            image_min = np.mean(image_min_stack, axis = 2)

                            # Image thresholding to remove background
                            image_min = image_min - config['image']['threshold'] * np.amax(image_min)
                            image_min[image_min < 0] = 0
                            self.image.emit(image_min)

                            # Append image to list
                            dset_append(data_set, 'real_calib_img', image_min)

                            # Set actuator back to bias voltage
                            voltages[i] = config['DM']['vol_bias']
                            
                        except Exception as e:
                            print(e)
                    else:

                        self.done.emit()

                # Close HDF5 file
                data_file.close()

                # Reset mirror
                self.mirror.Reset()

                # Calculate S-H spot centroids for each image in data list to get slopes
                if self.calc_cent:

                    self.slope_x, self.slope_y = acq_centroid(self.SB_settings, flag = 1)
                    self.slope_x -= np.mean(self.slope_x)
                    self.slope_y -= np.mean(self.slope_y)
                else:

                    self.done.emit()

                # Fill influence function matrix with acquired slopes
                if self.calc_inf:
                    
                    for i in range(self.actuator_num):

                        self.inf_matrix_slopes[:self.SB_settings['act_ref_cent_num'], i] = \
                            (self.slope_x[2 * i] - self.slope_x[2 * i + 1]) / (config['DM']['vol_max'] - config['DM']['vol_min'])
                        self.inf_matrix_slopes[self.SB_settings['act_ref_cent_num']:, i] = \
                            (self.slope_y[2 * i] - self.slope_y[2 * i + 1]) / (config['DM']['vol_max'] - config['DM']['vol_min'])             

                    # Calculate singular value decomposition of influence function matrix
                    u, s, vh = np.linalg.svd(self.inf_matrix_slopes, full_matrices = False)

                    # Calculate pseudo inverse of influence function matrix to get final control matrix
                    self.control_matrix_slopes = np.linalg.pinv(self.inf_matrix_slopes)

                    svd_check_slopes = np.dot(self.control_matrix_slopes, self.inf_matrix_slopes)

                    self.message.emit('\nDM calibration process finished.')

                else:

                    self.done.emit()

                prev2 = time.perf_counter()
                self.message.emit('\nTime for DM calibration process is: {} s.', (prev2 - prev1))     

            """
            Returns deformable mirror calibration information into self.mirror_info
            """ 
            if self.log:

                self.mirror_info['inf_matrix_slopes_SV'] = s
                self.mirror_info['inf_matrix_slopes'] = self.inf_matrix_slopes
                self.mirror_info['control_matrix_slopes'] = self.control_matrix_slopes
                self.mirror_info['calib_slope_x'] = self.slope_x
                self.mirror_info['calib_slope_y'] = self.slope_y
                self.mirror_info['svd_check_slopes'] = svd_check_slopes

                if self.debug:
                    self.mirror_info['act_pos_x'] = xc
                    self.mirror_info['act_pos_y'] = yc
                    self.mirror_info['act_diam'] = act_diam

                self.info.emit(self.mirror_info)
                self.write.emit()
            else:

                self.done.emit()

            # Finished calibrating deformable mirror and retrieving influence functions
            self.done.emit()

        except Exception as e:
            self.error.emit(e)
            raise

    @Slot(object)
    def stop(self):
        self.calibrate = False
        self.calc_cent = False
        self.calc_inf = False
        self.log = False
