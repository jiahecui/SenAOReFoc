import log
from config import config

logger = log.get_logger(__name__)


class SCANNER():
    """
    Scanner controller factory class, returns scanner instances of appropriate type.
    """
    def __init__(self):
        logger.info('Scanner factory class loaded.')

    @staticmethod
    def get(type = config['scanner']['SN']):
        if type.lower() == 'debug':
            scanner = SCANNER_dummy()
        else:
            scanner = None

        return scanner
        

class SCANNER_dummy():
    def __init__(self):
        logger.info('Dummy scanner loaded.')