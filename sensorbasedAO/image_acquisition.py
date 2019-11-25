import logging
import sys
import os
import imageio
import argparse
import time
import numpy as np

from ximea import xiapi

import log
from config import config
from sensor import SENSOR

logger = log.get_logger(__name__)

def acq_image(sensor, acq_mode = 0):
        """
        Acquires single image or image data list according to acq_mode, 0 for single image
        """
        # Create instance of dataimage array and data list to store image data
        data = []

        # Create instance of Ximea Image to store image data and metadata
        img = xiapi.Image()

        # Open device for centroiding instance 
        self.sensor.open_device_by_SN(config['camera']['SN'])

        # Start data acquisition for each frame
        print('Starting image acquisition...')
        self.sensor.start_acquisition()
        
        if acq_mode == 0:
            # Acquire one image and display
            try:
                # Get data and pass them from camera to img
                self.sensor.get_image(img, timeout = 25)

                # Create numpy array with data from camera, dimensions are determined by imgdataformats
                dataimage = img.get_image_data_numpy()
                
                # Bin pixels to fit on S-H viewer
                new_shape = 
                shape = (new_shape[0], array.shape[0] // new_shape[0], new_shape[1], array.shape[1] // new_shape[1])
        dataimage = array.reshape(shape).mean(-1).mean(1)
        
                dataimage = self.img_bin(dataimage, (self.sensor_width, self.sensor_height))

                # Display dataimage
                self.image.emit(dataimage)

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
                    self.sensor.get_image(img, timeout = 25)
                    prev2 = time.perf_counter()
                    print('Time for acquisition of frame {} is: {}'.format((i + 1), (prev2 - prev1)))

                    # Create numpy array with data from camera, dimensions are determined by imgdataformats
                    dataimages = img.get_image_data_numpy()

                    # Bin pixels to fit on S-H viewer
                    dataimages = self.img_bin(dataimages, (self.sensor_width, self.sensor_height))

                    # Display dataimage
                    self.image.emit(dataimages)
            
                    # Append dataimage to data list
                    data.append(dataimages)
            
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
        self.sensor.stop_acquisition()

        if acq_mode == 0:
            return dataimage
        elif acq_mode == 1:
            return data
        else:
            return None

    def img_bin(self, array, new_shape):
        """
        Bins numpy arrays to form new_shape by averaging pixels
        """
        shape = (new_shape[0], array.shape[0] // new_shape[0], new_shape[1], array.shape[1] // new_shape[1])
        new_array = array.reshape(shape).mean(-1).mean(1)

        return new_array