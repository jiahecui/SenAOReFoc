from PySide2.QtCore import QObject, Signal, Slot

import time
import numpy as np

import log
from config import config
from image_acquisition import acq_image

logger = log.get_logger(__name__)

class Acquisition(QObject):
    """
    Acquires SH images in different modes
    """
    # Signal class for starting an event
    start = Signal()

    # Signal class for exiting image acquisition event
    done = Signal()

    # Signal class for raising an error
    error = Signal(object)
    
    # Signal class for emitting a message in the message box
    message = Signal(object)

    # Signal class for displaying a SH spot image
    image = Signal(object)

    def __init__(self, sensor):

        # Get sensor instance
        self.sensor = sensor

        # Initialise sensor image parameters
        self.sensor_width = int(config['camera']['sensor_width'] // config['camera']['bin_factor'])
        self.sensor_height = int(config['camera']['sensor_height'] // config['camera']['bin_factor'])

        super().__init__()

    @Slot(object)
    def run0(self):
        try:
            # Set process flag
            self.loop = True

            self.message.emit('\nImage live acquisition started.')

            try:

                # Continuously acquire images while self.loop is true
                while self.loop:

                    # Acquire SH image
                    self._image = acq_image(self.sensor, self.sensor_height, self.sensor_width, acq_mode = 0)

                    # Image thresholding to remove background
                    self._image = self._image - config['image']['threshold'] * np.amax(self._image)
                    self._image[self._image < 0] = 0
                    self.image.emit(self._image)

                    time.sleep(config['camera']['sleep_time'])

            except Exception as e:
                print(e)

            self.message.emit('\nImage live acquisition finished.')
            
            self.done.emit()

        except Exception as e:
            self.error.emit(e)
            raise

    @Slot(object)
    def run1(self):
        try:
            # Set process flag
            self.loop = True

            self.message.emit('\nImage burst acquisition started.')

            try:

                # Burst acquire a specified number of images and display on GUI
                if self.loop:

                    # Acquire SH image
                    self._image_stack = acq_image(self.sensor, self.sensor_height, self.sensor_width, acq_mode = 1)

                    for i in range(np.shape(self._image_stack)[2]):

                        # Image thresholding to remove background
                        self._image_stack[:, :, i] = self._image_stack[:, :, i] - config['image']['threshold'] * np.amax(self._image_stack[:, :, i])
                        self._image_stack[self._image_stack[:, :, i] < 0] = 0
                        self.image.emit(self._image_stack[:, :, i])

                        time.sleep(config['camera']['sleep_time'])

            except Exception as e:
                print(e)

            self.message.emit('\nImage burst acquisition finished.')
            
            self.done.emit()

        except Exception as e:
            self.error.emit(e)
            raise

    @Slot(object)
    def run2(self):
        try:
            # Acquire single SH image
            self._image = acq_image(self.sensor, self.sensor_height, self.sensor_width, acq_mode = 0)

            # Image thresholding to remove background
            self._image = self._image - config['image']['threshold'] * np.amax(self._image)
            self._image[self._image < 0] = 0
            self.image.emit(self._image)           

            self.message.emit('\nSingle image acquired.')
            
            self.done.emit()

        except Exception as e:
            self.error.emit(e)
            raise

    @Slot(object)
    def stop(self):
        self.loop = False

            
