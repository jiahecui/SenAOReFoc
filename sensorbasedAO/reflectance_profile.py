import numpy as np
import random
import math

from config import config

def mirror_samp_sim():
    """
    Generates a mirror sample image
    """
    # Generate mirror sample image array
    mirror_samp_img = np.ones([config['reflect_prof']['samp_width'], config['reflect_prof']['samp_width']])

    return mirror_samp_img

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