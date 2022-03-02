from PySide2.QtCore import QObject, Signal, Slot

import numpy as np

import log
from config import config
from zernike import zern, zern_diff

logger = log.get_logger(__name__)

class Conversion(QObject):
    """
    Generates slope - zernike conversion matrix and zernike matrix for a given pupil shape
    """
    # Signal class for starting an event
    start = Signal()

    # Signal class for writing DM parameters into HDF5 file
    write = Signal()

    # Signal class for exiting slope - zernike conversion event
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
        self.SB_settings = settings

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.pupil_diam = config['search_block']['pupil_diam_0']
        elif config['DM']['DM_num'] == 1:
            self.pupil_diam = config['search_block']['pupil_diam_1']

        # Initialise conversion matrix information parameter
        self.conv_info = {}

        # Initialise zernike matrix and zernike derivative matrix
        self.zern_matrix = np.zeros([self.SB_settings['act_ref_cent_num'], config['AO']['recon_coeff_num']])
        self.diff_matrix = np.zeros([2 * self.SB_settings['act_ref_cent_num'], config['AO']['recon_coeff_num']]) 
        
        super().__init__()

    @Slot(object)
    def run(self):
        try:
            # Set process flags
            self.norm_coords = True
            self.calculate = True
            self.log = True

            # Start thread
            self.start.emit()

            if self.debug:

                print('')
                self.message.emit('\nExiting slope - Zernike conversion process.')

            else:

                """
                Get normalised reference coordinates in unit circle
                """
                if self.norm_coords:

                    # Get rescale factor between normalised coordinates and actual coordinates
                    self.rescale = 2 / (self.pupil_diam * 1e3 / self.SB_settings['pixel_size'])

                    # Get normalised search block radius
                    self.norm_rad = self.SB_settings['SB_rad'] * self.rescale

                    # Get size of each individual element in unit circle
                    self.elem_size = self.SB_settings['SB_diam'] / config['search_block']['div_elem'] * self.rescale
                    
                    # Get reference centroid coordinates for unit circle
                    self.norm_ref_cent_coord_x = (self.SB_settings['act_ref_cent_coord_x'] - \
                        self.SB_settings['sensor_width'] // 2 - self.SB_settings['act_SB_offset_x']) * self.rescale
                    self.norm_ref_cent_coord_y = (self.SB_settings['act_ref_cent_coord_y'] - \
                        self.SB_settings['sensor_height'] // 2 - self.SB_settings['act_SB_offset_y']) * self.rescale

                    # Take account of odd number of relays and mirrors between DM and lenslet
                    if config['relay']['mirror_odd']:
                        self.norm_ref_cent_coord_x = -self.norm_ref_cent_coord_x

                    if config['relay']['relay_odd']:
                        self.norm_ref_cent_coord_x = -self.norm_ref_cent_coord_x
                        self.norm_ref_cent_coord_y = -self.norm_ref_cent_coord_y

                else:

                    self.done.emit()

                """
                Get normalised coordinates for each individual element in search block to calculate zernike matrix and conversion matrix
                """
                if self.calculate:
                    
                    for i in range(self.SB_settings['act_ref_cent_num']):

                        # Get reference centroid coords of each element
                        elem_ref_cent_coord_x = np.arange(self.norm_ref_cent_coord_x[i] - self.norm_rad + self.elem_size / 2, \
                            self.norm_ref_cent_coord_x[i] + self.norm_rad, self.elem_size)
                        elem_ref_cent_coord_y = np.arange(self.norm_ref_cent_coord_y[i] - self.norm_rad + self.elem_size / 2, \
                            self.norm_ref_cent_coord_y[i] + self.norm_rad, self.elem_size)

                        elem_ref_cent_coord_xx, elem_ref_cent_coord_yy = np.meshgrid(elem_ref_cent_coord_x, elem_ref_cent_coord_y)

                        # Get averaged x and y values and derivatives of the jth Zernike polynomial to fill zernike matrix and zernike derivative matrix
                        for j in range(config['AO']['recon_coeff_num']):
                            
                            self.zern_matrix[i, j] = zern(elem_ref_cent_coord_xx, elem_ref_cent_coord_yy, j + 1)

                            self.diff_matrix[i, j] = zern_diff(elem_ref_cent_coord_xx, elem_ref_cent_coord_yy, j + 1, True)
                            self.diff_matrix[i + self.SB_settings['act_ref_cent_num'], j] = zern_diff(elem_ref_cent_coord_xx, elem_ref_cent_coord_yy, j + 1, False)
                    
                    # Take beam diameter on lenslet, pixel size, and lenslet focal length into account
                    self.diff_matrix = self.diff_matrix * 2 * config['lenslet']['lenslet_focal_length'] / (self.pupil_diam * 1e3 * self.SB_settings['pixel_size'])
                    
                    # Get singular value decomposition of zernike derivative matrix
                    u, s, vh = np.linalg.svd(self.diff_matrix, full_matrices = False)
                    
                    # Calculate pseudo inverse of zernike derivative matrix to get conversion matrix
                    self.conv_matrix = np.linalg.pinv(self.diff_matrix)

                    svd_check_conv = np.dot(self.conv_matrix, self.diff_matrix)

                    # Convert zern_matrix from radians to um
                    self.zern_matrix = self.zern_matrix * config['AO']['lambda'] / (2 * np.pi)   

                    self.message.emit('\nZernike matrix and slope - Zernike conversion matrix retrieved.')
                else:

                    self.done.emit()

            """
            Returns zernike matrix and slope - zernike conversion matrix information into self.conv_info
            """ 
            if self.log and not self.debug:

                self.conv_info['norm_ref_cent_coord_x'] = self.norm_ref_cent_coord_x
                self.conv_info['norm_ref_cent_coord_y'] = self.norm_ref_cent_coord_y
                self.conv_info['conv_matrix_SV'] = s
                self.conv_info['zern_matrix'] = self.zern_matrix
                self.conv_info['diff_matrix'] = self.diff_matrix
                self.conv_info['conv_matrix'] = self.conv_matrix
                self.conv_info['svd_check_conv'] = svd_check_conv

                self.info.emit(self.conv_info)
                self.write.emit()
            else:

                self.done.emit()

            # Finished retrieving slope - zernike conversion matrix
            self.done.emit()

        except Exception as e:
            self.error.emit(e)
            raise

    @Slot(object)
    def stop(self):
        self.norm_coords = False
        self.calculate = False
        self.log = False