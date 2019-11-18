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
from gui.main import Main
from SB_geometry import Setup_SB
from centroiding import Centroiding

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
        """
        Setup search block geometry and get reference centroids
        """
        # Create SB worker and thread
        SB_thread = QThread()
        SB_thread.setObjectName('SB_thread')
        SB_worker = Setup_SB()
        SB_worker.moveToThread(SB_thread)
        
        # Connect to signals
        SB_thread.started.connect(SB_worker.run)
        SB_worker.layer.connect(lambda obj: self.main.update_image(obj))
        SB_worker.info.connect(lambda obj: self.handle_SB_info(obj))
        SB_worker.error.connect(lambda obj: self.handle_error(obj))
        SB_worker.done.connect(self.handle_SB_done)

        # Store SB worker and thread
        self.workers['SB_worker'] = SB_worker
        self.threads['SB_thread'] = SB_thread

        # Start SB thread
        SB_thread.start()

    def get_centroids(self, sensor, SB_info):
        """
        Get actual centroids of S-H spots
        """
        print('Starting to get centroids')
        # Create centroiding worker and thread
        cent_thread = QThread()
        cent_thread.setObjectName('cent_thread')
        cent_worker = Centroiding(sensor, SB_info)
        cent_worker.moveToThread(cent_thread)

        # Connect to signals
        cent_thread.started.connect(cent_worker.run)
        cent_worker.image.connect(lambda obj: self.main.update_image(obj))
        cent_worker.error.connect(lambda obj: self.handle_error(obj))
        # cent_worker.done.connect()
        # cent_worker.info.connect()

        # Store centroiding worker and thread
        self.workers['cent_worker'] = cent_worker
        self.threads['cent_thread'] = cent_thread

        # Start SB thread
        cent_thread.start()

    #========== Signal handlers ==========#
    def handle_SB_info(self, obj):
        """
        Handle search block geometry and reference centroid information
        """
        self.SB_info = obj

    def handle_error(self, error):
        """
        Handle errors from threads
        """
        raise(RuntimeError(error))
        
    def handle_SB_done(self):
        """
        Handle end of search block geometry setup
        """
        self.threads['SB_thread'].quit()
        self.threads['SB_thread'].wait()
        self.main.ui.initialiseBtn.setChecked(False)
        self.get_centroids(self.devices['sensor'], self.SB_info)


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

    app = App(debug = False)

    sys.exit(app.exec_())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Sensorbased AO gui')
    parser.add_argument("-d", "--debug", help = 'debug mode',
                        action = "store_true")
    args = parser.parse_args()

    if args.debug:  
        debug()
    else:
        main()