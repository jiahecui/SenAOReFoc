import numpy as np
import math
from config import config
from zernike import zern_gen

def zern_phase(settings, zern_array):
    """
    Generates zernike phase map by superpositioning different amounts of zernike modes according to zern_array
    """
    # Choose working DM along with its parameters
    if config['DM']['DM_num'] == 0:
        pupil_diam = config['search_block']['pupil_diam_0']
    elif config['DM']['DM_num'] == 1:
        pupil_diam = config['search_block']['pupil_diam_1']

    # Get the number of pixels along the diameter of the pupil
    pupil_diam_pixel = int(pupil_diam * 1e3 / settings['pixel_size'])

    # Initialise phase map
    zern_phase_img = np.zeros([pupil_diam_pixel, pupil_diam_pixel])

    # Make sure there is an even number of pixels along the diameter of the pupil
    if pupil_diam_pixel % 2 == 1:
        pupil_diam_pixel -= 1
    else:
        pass

    # Get x,y coordinates of each pixel
    xx, yy = np.meshgrid(np.arange(-(pupil_diam_pixel // 2), pupil_diam_pixel // 2), \
        np.arange(-(pupil_diam_pixel // 2), pupil_diam_pixel // 2))

    # Normalise x,y coordinates
    xx_norm = xx / (pupil_diam_pixel // 2)
    yy_norm = yy / (pupil_diam_pixel // 2)

    print('Starting to generate zernike phase map...')

    # Get zernike phase map
    for i in range(len(zern_array)):
        zern_phase_img += zern_array[i] * zern_gen(xx_norm, yy_norm, i + 1)

    # Pad image to same dimension as sensor size
    zern_phase_img_pad = np.pad(zern_phase_img, (math.ceil((settings['sensor_height'] - np.shape(zern_phase_img)[0]) / 2),\
        math.ceil((settings['sensor_width'] - np.shape(zern_phase_img)[1]) / 2)), 'constant', constant_values = (0, 0))

    print('Zernike phase map generated.')
    
    return zern_phase_img_pad
