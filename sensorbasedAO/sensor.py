import os
import sys
import time

from ximea import xiapi
# from devwraps.ueye import uEye

import log
from config import config

logger = log.get_logger(__name__)


class SENSOR_XIMEA(xiapi.Camera):
    """
    Create an instance of the Camera class to access its methods, attributes, and the target device at serial port.
    """
    def __init__(self):
        # Create Camera instance
        self.sensor = xiapi.Camera()

        # Open camera by serial number
        self.sensor.open_device_by_SN(config['camera']['SN'])
        
        # Set camera image format
        self.sensor.set_imgdataformat(config['camera']['dataformat'])

        # Set camera exposure and gain preference
        if config['camera']['auto_gain'] == 0:
            self.sensor.set_exposure(config['camera']['exposure'])
        else:
            self.sensor.enable_aeag()
            self.sensor.set_exp_priority(config['camera']['exp_priority'])
            self.sensor.set_ae_max_limit(config['camera']['ae_max_limit'])

        # Set camera frame acquisition mode       
        if config['camera']['burst_mode'] == 1:
            self.sensor.set_trigger_selector('XI_TRG_SEL_FRAME_BURST_START')
            self.sensor.set_acq_frame_burst_count(config['camera']['burst_frames'])
        else:
            self.sensor.set_acq_timing_mode(config['camera']['acq_timing_mode'])
            self.sensor.set_framerate(config['camera']['frame_rate'])

        # Set camera trigger mode
        self.sensor.set_trigger_source(config['camera']['trigger_source'])

        super().__init__()

# class SENSOR_IDS(uEye):
#     """
#     Create an instance of the uEye class to access its methods, attributes, and the target device at serial port.
#     """
#     def __init__(self):

#         # Create uEye instance
#         self.sensor = uEye()

#         # Get uEye device
#         self.sensor.get_devices()

#         # Open uEye device
#         self.sensor.open(config['camera']['SN'])

#         # Get sensor shape upon initialisation
#         shape = self.sensor.shape()
#         print('Shape of sensor upon initialisation is', shape)

#         # Get sensor frame rate upon initialisation and frame rate range 
#         # frame_rate = self.sensor.get_framerate()
#         # frame_rate_range = self.sensor.get_framerate_range()
#         # print('Frame rate of sensor upon initialisation is', frame_rate)
#         # print('Frame rate range of sensor is', frame_rate_range)

#         # Set sensor exposure time (ms)
#         self.sensor.set_exposure(10.0)
#         # exp_time = self.sensor.get_exposure()
#         # print('Exposure time of sensor is', exp_time)

#         # Set sensor auto gain
#         self.sensor.set_auto_gain(1)
#         # auto_gain = self.sensor.get_auto_gain()
#         # print('Auto gain of sensor is', auto_gain)

#         super().__init__()


class SENSOR():
    """
    Sensor controller factory class, returns sensor instances of appropriate type.
    """
    def __init__(self):
        logger.info('Sensor factory class loaded')

    @staticmethod
    def get(type = config['camera']['SN']):
        if type.lower() == '26883050':
            try:
                sensor = SENSOR_XIMEA()
            except:
                logger.warning('Unable to load Ximea camera')
        elif type.lower() == '4002885218':
            try:
                sensor = SENSOR_IDS()
            except:
                logger.warning('Unable to load IDS sensor')
        elif type.lower() == 'debug':
            sensor = SENSOR_dummy()
        else:
            sensor = None

        return sensor


class SENSOR_dummy():
    def __init__(self):
        logger.info('Dummy sensor loaded')


if __name__ == '__main__':
    main()



    


