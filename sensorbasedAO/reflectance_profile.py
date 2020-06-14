import numpy as np
import random
import math
import h5py

from config import config

def bead_samp_sim():
    """
    Generates a bead sample image with Gaussian profile beads at random positions
    """
    try:
        # Initialise bead sample image array
        bead_samp_img = np.ones([config['reflect_prof']['samp_width'], config['reflect_prof']['samp_width']]) * config['reflect_prof']['background']

        # Generate meshgrid of image coordinate arrays
        xx, yy = np.meshgrid(np.arange(0, config['reflect_prof']['samp_width']), np.arange(0, config['reflect_prof']['samp_width']))

        # Generate random centre coordinates for Gaussian profile beads
        xc, yc = (np.random.randint(2, config['reflect_prof']['samp_width'] - 1, config['reflect_prof']['bead_num']) for i in range(2))

        print('Starting to generate bead sample image...')

        # Generate bead sample image
        for i in range(config['reflect_prof']['bead_num']):
            bead_samp_img += (255 * np.exp( - ((xx - xc[i]) ** 2 + (yy - yc[i]) ** 2) / (2 * config['reflect_prof']['bead_sigma'] ** 2))).astype(np.uint8)

        bead_samp_img = bead_samp_img / np.amax(bead_samp_img)

        print('Bead sample image generated.')
    except Exception as e:
        print(e)

    return bead_samp_img

def get_samp_sim(sample = 0):
    """
    Function to load mirror sample or .mat file sample profiles

    Args:
        sample0 - mirror sample
        sample1 - a vertical line of ones two pixels wide through the centre
        sample2 - a rad = 5 disk of ones in the centre
        sample3 - a rad = 5 disk of ones in the second quad
        sample4 - a rad = 5 disk of ones on the left
        sample5 - two rad = 5 disks of ones, one in the second quad, another in the fourth quad
        sample6 - a 4-pixel dot in the centre
        sample7 - a rad = 2 disk of ones in the centre
        sample8 - a rad = 3 disk of ones in the centre
        sample9 - a rad = 4 disk of ones in the centre
        sample10 - a rad = 6 disk of ones in the centre
        sample11 - a rad = 7 disk of ones in the centre
        sample12 - a rad = 8 disk of ones in the centre
        sample13 - a rad = 12 disk of ones in the centre
        sample14 - a rad = 12 disk of ones shifted 6 pixels to the left
        sample15 - a rad = 12 disk of ones shifted 12 pixels to the left
        sample16 - a rad = 12 disk of ones shifted 18 pixels to the left
        sample17 - a rad = 24 disk of ones in the centre
        sample18 - a rad = 5 disk of ones shifted 2 pixels to the left
        sample19 - a rad = 5 disk of ones shifted 7 pixels to the left
        sample20 - a rad = 3 disk of ones shifted 3 pixels to the left
        sample21 - a rad = 3 disk of ones shifted 5 pixels to the left
        sample22 - a 4-pixel dot shifted 1 pixel to the left
        sample23 - a 4-pixel dot shifted 2 pixels to the left
        sample24 - a 4-pixel dot shifted 3 pixels to the left
        sample25 - random rad = 12 and rad = 24 disks of ones scattered
        sample26 - disks of all sizes arranged in one image
        sample27 - sample 25 with structure reflectivity 0.8 and tissue reflectivity 1
        sample28 - sample 25 with structure reflectivity 0.5 and tissue reflectivity 1
        sample29 - sample 25 with structure reflectivity 0 and tissue reflectivity 1
        sample30 - sample 17 with structure reflectivity 0 and tissue reflectivity 1
        sample31 - a vertical line of ones 4 pixels wide through the centre
        sample32 - a vertical line of ones 6 pixels wide through the centre
        sample33 - a vertical line of ones 12 pixels wide through the centre
        sample34 - a 12 pixel wide cross of ones in the centre
    """
    if sample == 0:
        sample_img = np.ones([config['reflect_prof']['obj_grid_size'], config['reflect_prof']['obj_grid_size']])
    else:
        sample_img = h5py.File('sensorbasedAO/Sample' + str(sample) + '.mat','r').get('sample' + str(sample))

    sample_img = np.array(sample_img)

    return sample_img
