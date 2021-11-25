
from alpao.Lib64 import asdk  # Use alpao.Lib for 32-bit applications and alpao.Lib64 for 64-bit applications

import log
from config import config

logger = log.get_logger(__name__)


class MIRROR_ALPAO(asdk.DM):
    """
    Creates an instance of the DM class to access its methods, attributes, and the target device at serial port.
    """
    def __init__(self, serialNumber):
        # Create Mirror instance
        self.mirror = asdk.DM(serialNumber)

        # Get number of actuators for Mirror instance
        nbAct = int(self.mirror.Get('NBOfActuator')) 
        print('Number of actuators for ' + serialNumber + ': ' + str(nbAct))
    
        # Reset Mirror instance
        self.mirror.Reset()

        super().__init__(serialNumber)


class MIRROR():
    """
    Mirror controller factory class, returns mirror instances of appropriate type.
    """
    def __init__(self):
        logger.info('Mirror factory class loaded.')

    @staticmethod
    def get(type = config['DM']['SN']):
        if type.lower() == 'hsdm69-15-014':
            try:
                mirror = MIRROR_ALPAO(type)
            except:
                logger.warning('Unable to load Alpao DM.')
        elif type.lower() == 'debug':
            mirror = MIRROR_dummy()
        else:
            mirror = None

        return mirror


class MIRROR_dummy():
    def __init__(self):
        logger.info('Dummy mirror loaded.')
