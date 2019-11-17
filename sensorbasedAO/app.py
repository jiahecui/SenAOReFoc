from PySide2.QtWidgets import QApplication, QStyleFactory, QMainWindow, QFileDialog, QDialog, QVBoxLayout, QWidget
from PySide2.QtCore import QThread, QObject, Slot, Signal, QSize, QTimer

import logging
import sys
import os
import imageio
import argparse
import math
import time
import click
import PIL.Image
import numpy as np

from datetime import datetime

# from alpao import asdk
from ximea import xiapi

import log
from config import config
from sensor import SENSOR

logger = log.get_logger(__name__)

class App(QApplication):
    """
    The main application

    Based on the QApplication class. On instantiation, sets up GUI and event handlers, and displays the GUI.
    Also instantiates devices, such as S-H sensor and DM and adds as attributes.

    Args:
        parent(: obj: `QObject`, optional): parent window

    Example:
        Here is a simple example of using the class: :

            app = App(sys.argv)
            sys.exit(app.exec_())
    """
    def __init__(self, debug = False):
        QThread.currentThread().setObjectName('app')
        
        super().__init__()

        self.setStyle('fusion')

        self.debug = debug

        # Initialise instance variables
        self.image_temp = np.zeros([100, 100])

        # Initialise workers and threads
        self.workers = {}
        self.threads = {}

        # Add devices
        self.devices = {}
        self.add_devices()

        # Open main GUI window
        self.main = Main(self, debug = debug)
        self.main.show()

    def add_devices(self):
        """
        Add hardware devices to a dictionary
        """
        # Add S-H sensor
        if self.debug:
            sensor = SENSOR.get('debug')
        else:
            try:
                sensor = SENSOR.get(config['camera']['SN'])
                print('Sensor load success')
            except Exception as e:
                logger.warning('Sensor load error', e)
                sensor = None

        self.devices['sensor'] = sensor

    def setup_SB(self):


def debug():
    logger.setLevel(logging.DEBUG)
    handler_stream = logging.StreamHandler()
    handler_stream.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_stream)
    logger.info('Started sensorbased AO app in debug mode')

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
    logger.info('Started sensorbased AO app')

    SB = Setup_SB(debug = False)
    SB.register_SB()
    SB.make_reference_SB()
    SB.get_SB_geometry()
    SB_info = SB.get_SB_info()

    try:
        sensor = SENSOR.get(config['camera']['SN'])
        print('Sensor load success')
    except Exception as e:
        logger.warning('Sensor load error', e)
        sensor = None

    Cent = Centroiding(sensor, SB_info)
    Cent.get_SB_position()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Sensorbased AO gui')
    parser.add_argument("-d", "--debug", help = 'debug mode',
                        action = "store_true")
    args = parser.parse_args()

    if args.debug:  
        debug()
    else:
        main()