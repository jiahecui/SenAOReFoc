import os
import sys
import time

import mtidevice
from mtidevice import MTIError, MTIAxes, MTIParam, MTIDataMode, MTISync, MTIDataFormat, MTIAvailableDevices

import log
from config import config

logger = log.get_logger(__name__)

class MEMS_MIRRORCLE(mtidevice.MTIDevice):
    """
    Creates an instance of the MTIDevice class to access its methods and attributes, locates the target device at serial port,
    loads parameters provided in mtidevice.ini file, and updates the controller
    """
    def __init__(self, serialNumber):
        # Create MEMS instance, locate serial port and connect to device
        self.mtidevice = mtidevice.MTIDevice()
        table = self.mtidevice.GetAvailableDevices()

        if table.NumDevices == 0:
            print('There are no devices available.')
            return None

        self.mtidevice.ListAvailableDevices(table)
        portnumber = config['scanner']['portnumber']
        
        if os.name == 'nt':
            portName = 'COM' + portnumber
        else:
            portName = '/dev/ttyUSB' + portnumber

        self.mtidevice.ConnectDevice(portName)

        # Initialise controller parameters
        params = self.mtidevice.GetDeviceParams()
        params.VdifferenceMax = 159
        params.HardwareFilterBw = 500
        params.Vbias = 80
        params.SampleRate = 20000
        self.mtidevice.SetDeviceParams(params)

        params_temp = self.mtidevice.GetDeviceParams()

        # Set controller data mode
        self.mtidevice.ResetDevicePosition()
        self.mtidevice.StartDataStream()

        # Turn the MEMS controller on
        self.mtidevice.SetDeviceParam(MTIParam.MEMSDriverEnable, True)

        super().__init__() 


class SCANNER():
    """
    Scanner controller factory class, returns scanner instances of appropriate type.
    """
    def __init__(self):
        logger.info('Scanner factory class loaded.')

    @staticmethod
    def get(type = config['scanner']['SN']):
        if type.lower() == 's31155':
            try:
                scanner = MEMS_MIRRORCLE(type)
            except:
                logger.warning('Unable to load Mirrorcle mirror.')
        elif type.lower() == 'debug':
            scanner = SCANNER_dummy()
        else:
            scanner = None

        return scanner
        

class SCANNER_dummy():
    def __init__(self):
        logger.info('Dummy scanner loaded.')
    

if __name__ == '__main__':
    main()
