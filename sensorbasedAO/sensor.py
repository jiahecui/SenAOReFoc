import os
import sys
import time
import numpy as np

import ximea

import log
from config import config

logger = log.get_logger(__name__)


class SENSOR_XIMEA(ximea.xiapi):
    def __init__(self):
        """
        Create an instance of the Camera class to access its methods, attributes, and the target device at serial port.
        """
        self.sensor = xiapi.Camera()

        self.sensor.open_device_by_SN(config['camera']['SN'])
        print('Device type is %s' % cam.get_device_type())
        print('Device model ID is %s' % cam.get_device_model_id())
        print('Device name is %s' % cam.get_device_name())

        try:
            self.sensor.set_imgdataformat(config['camera']['dataformat'])
        except xiapi.Xi_error as err:
            print('Error code is:', err.status)

        if config['camera']['auto_gain'] == 0:
            self.sensor.set_exposure(config['camera']['exposure'])
        else:
            self.sensor.set_aeag(config['camera']['aeag'])
            self.sensor.set_exp_priority(config['camera']['exp_priority'])
            self.sensor.set_ae_max_limit(config['camera']['ae_max_limit'])
        print('2')
        self.sensor.set_trigger_source(config['camera']['trg_source'])
        self.sensor.set_trigger_overlap(config['camera']['trg_overlap'])
        print('3')
        if config['camera']['burst_mode']:
            self.camera.set_trigger_selector('XI_TRG_SEL_FRAME_BURST_START')
            self.camera.set_acq_frame_burst_count(config['camera']['burst_frames'])
        else:
            self.camera.set_trigger_selector('XI_TRG_SEL_ACQUISITION_START')
        print('4')
        self.sensor.set_acq_timing_mode(config['camera']['acq_timing_mode'])
        self.sensor.set_framerate(config['camera']['framerate'])
        print('5')

        super().__init__()


class SENSOR():
    """
    Sensor controller factory class, returns sensor instances of appropriate type.
    """
    @staticmethod
    def get(type = '26883050'):
        if type.lower() == '26883050':
            try:
                sensor =  SENSOR_XIMEA()
                print('Ximea camera loaded')
            except:
                logger.warning('Unable to load Ximea camera')
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



    


