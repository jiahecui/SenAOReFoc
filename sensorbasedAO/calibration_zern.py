from PySide2.QtCore import QObject, Signal, Slot
import numpy as np

import log
from config import config

logger = log.get_logger(__name__)

class Calibration_Zern(QObject):
    """
    Retrieves control matrix via zernikes using slopes acquired during calibration via slopes
    """
    # Signal class for starting an event
    start = Signal()

    # Signal class for writing DM parameters into HDF5 file
    write = Signal()

    # Signal class for exiting DM Zernike calibration event
    done = Signal()

    # Signal class for emitting a message in the message box
    message = Signal(object)

    # Signal class for raising an error
    error = Signal(object)

    # Signal class for updating DM control parameters
    info = Signal(object)

    def __init__(self, settings, debug = False):

        # Get debug status
        self.debug = debug

        # Get search block settings
        self.SB_settings = settings['SB_info']

        # Get mirror settings
        self.mirror_settings = settings['mirror_info']

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']

        # Initialise deformable mirror information parameter
        self.mirror_info = {}

        # Initialise influence function matrix
        self.inf_matrix_zern = np.zeros([config['AO']['recon_coeff_num'], self.actuator_num])
        
        super().__init__()

    @Slot(object)
    def run(self):
        try:
            # Set process flags
            self.calc_inf = True
            self.log = True

            # Start thread
            self.start.emit()

            if self.debug:

                print('')
                self.message.emit('\nExiting Zernike control matrix calibration process.')

            else:

                """
                Calculate individual zernike coefficients for max / min voltage of each actuator using preacquired calibration slopes
                and slope - zernike conversion matrix to create zernike influence function and control matrix
                """
                if self.calc_inf:

                    # Get calibration slopes and slope - zernike conversion matrix
                    self.slope_x = self.mirror_settings['calib_slope_x']
                    self.slope_y = self.mirror_settings['calib_slope_y']
                    self.conv_matrix =  self.mirror_settings['conv_matrix']

                    # Convert slopes list to numpy array
                    (self.slope_x, self.slope_y) = map(np.array, (self.slope_x, self.slope_y))

                    # Concatenate x, y slopes matrix
                    self.slope = np.concatenate((self.slope_x.T, self.slope_y.T), axis = 0)

                    # Fill influence function matrix by multiplying each column in slopes matrix with the conversion matrix
                    for i in range(self.actuator_num):

                        self.inf_matrix_zern[:, i] = \
                            np.dot(self.conv_matrix, (self.slope[:, 2 * i] - self.slope[:, 2 * i + 1])) \
                                / (config['DM']['vol_max'] - config['DM']['vol_min'])
                
                    # Get singular value decomposition of influence function matrix
                    u, s, vh = np.linalg.svd(self.inf_matrix_zern, full_matrices = False)

                    # print('u: {}, s: {}, vh: {}'.format(u, s, vh))
                    # print('The shapes of u, s, and vh are: {}, {}, and {}'.format(np.shape(u), np.shape(s), np.shape(vh)))
                    
                    # Calculate pseudo inverse of influence function matrix to get final control matrix
                    self.control_matrix_zern = np.linalg.pinv(self.inf_matrix_zern, rcond = 0.01)

                    svd_check_zern = np.dot(self.inf_matrix_zern, self.control_matrix_zern)

                    self.message.emit('\nZernike control matrix retrieved.')
                else:

                    self.done.emit()

            """
            Returns zernike calibration information into self.mirror_info
            """ 
            if self.log and not self.debug:

                self.mirror_info['inf_matrix_zern_SV'] = s
                self.mirror_info['inf_matrix_zern'] = self.inf_matrix_zern
                self.mirror_info['control_matrix_zern'] = self.control_matrix_zern
                self.mirror_info['svd_check_zern'] = svd_check_zern

                self.info.emit(self.mirror_info)
                self.write.emit()
            else:

                self.done.emit()

            # Finished retrieving zernike influence function and control matrix
            self.done.emit()

        except Exception as e:
            self.error.emit(e)
            raise

    @Slot(object)
    def stop(self):
        self.calc_inf = False
        self.log = False
