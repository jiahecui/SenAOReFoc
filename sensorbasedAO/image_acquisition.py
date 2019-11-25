import sys
import os
import argparse
import time
import numpy as np

from ximea import xiapi

import log
from config import config
from sensor import SENSOR

logger = log.get_logger(__name__)

def acq_image(sensor, width, height, acq_mode = 0):
    """
    Acquires single image or image data list according to acq_mode, 0 for single image, 1 for a sequence of images
    """
    # Create instance of dataimage array and data list to store image data
    data = []

    # Create instance of Ximea Image to store image data and metadata
    img = xiapi.Image()

    # Open device for centroiding instance 
    sensor.open_device_by_SN(config['camera']['SN'])

    # Start data acquisition for each frame
    print('Starting image acquisition...')
    sensor.start_acquisition()
    
    if acq_mode == 0:
        # Acquire one image
        try:
            # Get data and pass them from camera to img
            sensor.get_image(img, timeout = 25)

            # Create numpy array with data from camera, dimensions are determined by imgdataformats
            dataimage = img.get_image_data_numpy()
            
            # Bin numpy arrays by averaging pixels to fit on S-H viewer
            shape = (width, dataimage.shape[0] // width, height, dataimage.shape[1] // height)
            dataimage = dataimage.reshape(shape).mean(-1).mean(1)

        except xiapi.Xi_error as err:
            if err.status == 10:
                print('Timeout error occurred.')
            else:
                raise

    elif acq_mode == 1:
        # Acquire a sequence of images and append to data list
        for i in range(config['camera']['frame_ave_num']):
            prev1 = time.perf_counter()

            try:
                # Get data and pass them from camera to img
                sensor.get_image(img, timeout = 25)
                prev2 = time.perf_counter()
                print('Time for acquisition of frame {} is: {}'.format((i + 1), (prev2 - prev1)))

                # Create numpy array with data from camera, dimensions are determined by imgdataformats
                dataimage = img.get_image_data_numpy()

                # Bin numpy arrays by averaging pixels to fit on S-H viewer
                shape = (width, dataimage.shape[0] // width, height, dataimage.shape[1] // height)
                dataimage = dataimage.reshape(shape).mean(-1).mean(1)

                # Append dataimage to data list
                data.append(dataimage)
        
            except xiapi.Xi_error as err:
                if err.status == 10:
                    print('Timeout error occurred.')
                else:
                    raise

            prev3 = time.perf_counter()
            print('Time for acquisition of loop {} is: {}'.format((i + 1), (prev3 - prev1)))

        print('Length of data list is:', len(data))

    # Stop data acquisition
    print('Stopping image acquisition...')
    sensor.stop_acquisition()

    if acq_mode == 0:
        return dataimage
    elif acq_mode == 1:
        return data
    else:
        return None