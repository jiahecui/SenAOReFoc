import logging
import sys
import os
import imageio
import argparse
import numpy as np

from datetime import datetime

from ximea import xiapi
from alpao import asdk

import log
from config import config

logger = log.get_logger()



"""
def debug():
    logger.setLevel(logging.DEBUG)
    handler_stream = logging.StreamHandler()
    handler_stream.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_stream)
    logger.info('Started sensor-based AO app in debug mode')

    app = App(debug=True)

    sys.exit(app.exec_())

def main():
    logger.setLevel(logging.DEBUG)
    handler_stream = logging.StreamHandler()  
    handler_stream.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_stream)
    logger.info('Started sensor-based AO app')

    app = App(debug=False)

    sys.exit(app.exec_())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sensor-based AO gui')
    parser.add_argument("-d", "--debug", help='debug mode',
                        action="store_true")
    args = parser.parse_args()

    if args.debug:
        debug()
    else:
        main()
"""