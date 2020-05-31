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
    """
    if sample == 0:
        sample_img = np.ones([config['reflect_prof']['obj_grid_size'], config['reflect_prof']['obj_grid_size']])
    elif sample == 1:
        sample_img = h5py.File('sensorbasedAO/Sample1.mat','r').get('sample1')
    elif sample == 2:
        sample_img = h5py.File('sensorbasedAO/Sample2.mat','r').get('sample2')
    elif sample == 3:
        sample_img = h5py.File('sensorbasedAO/Sample3.mat','r').get('sample3')
    elif sample == 4:
        sample_img = h5py.File('sensorbasedAO/Sample4.mat','r').get('sample4')
    elif sample == 5:
        sample_img = h5py.File('sensorbasedAO/Sample5.mat','r').get('sample5')
    elif sample == 6:
        sample_img = h5py.File('sensorbasedAO/Sample6.mat','r').get('sample6')
    elif sample == 7:
        sample_img = h5py.File('sensorbasedAO/Sample7.mat','r').get('sample7')
    elif sample == 8:
        sample_img = h5py.File('sensorbasedAO/Sample8.mat','r').get('sample8')
    elif sample == 9:
        sample_img = h5py.File('sensorbasedAO/Sample9.mat','r').get('sample9')
    elif sample == 10:
        sample_img = h5py.File('sensorbasedAO/Sample10.mat','r').get('sample10')
    elif sample == 11:
        sample_img = h5py.File('sensorbasedAO/Sample11.mat','r').get('sample11')
    elif sample == 12:
        sample_img = h5py.File('sensorbasedAO/Sample12.mat','r').get('sample12')
    elif sample == 13:
        sample_img = h5py.File('sensorbasedAO/Sample13.mat','r').get('sample13')
    elif sample == 14:
        sample_img = h5py.File('sensorbasedAO/Sample14.mat','r').get('sample14')
    elif sample == 15:
        sample_img = h5py.File('sensorbasedAO/Sample15.mat','r').get('sample15')
    elif sample == 16:
        sample_img = h5py.File('sensorbasedAO/Sample16.mat','r').get('sample16')

    sample_img = np.array(sample_img)

    return sample_img
