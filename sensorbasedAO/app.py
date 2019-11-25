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

# from alpao.Lib import asdk  # Use alpao.Lib for 32-bit applications and alpao.Lib64 for 64-bit applications
from ximea import xiapi

import log
from config import config
from sensor import SENSOR
# from mirror import MIRROR
from gui.main import Main
from SB_geometry import Setup_SB
from centroiding import Centroiding
from calibration import Calibration

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

        # Initialise search block info dictionary
        self.SB_info = {}

        # Initialise deformable mirror info dictionary
        self.mirror_info = {}

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

        # Add deformable mirror
        if self.debug:
            mirror = MIRROR.get('debug')
        else:
            try:
                mirror = MIRROR.get(config['DM']['SN'])
                print('Mirror load success')
            except Exception as e:
                logger.warning('Mirror load error', e)
                mirror = None

        self.devices['mirror'] = mirror

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
        SB_worker.layer.connect(lambda obj: self.handle_layer_disp(obj))
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
        # Create centroiding worker and thread
        cent_thread = QThread()
        cent_thread.setObjectName('cent_thread')
        cent_worker = Centroiding(sensor, SB_info)
        cent_worker.moveToThread(cent_thread)

        # Connect to signals
        cent_thread.started.connect(cent_worker.run)
        cent_worker.layer.connect(lambda obj: self.handle_layer_disp(obj))
        cent_worker.image.connect(lambda obj: self.handle_image_disp(obj))
        cent_worker.info.connect(lambda obj: self.handle_SB_info(obj))
        cent_worker.error.connect(lambda obj: self.handle_error(obj))
        cent_worker.done.connect(self.handle_cent_done)

        # Store centroiding worker and thread
        self.workers['cent_worker'] = cent_worker
        self.threads['cent_thread'] = cent_thread

        # Start centroiding thread
        cent_thread.start()

    def calibrate_DM(self, sensor, mirror, SB_info):
        """
        Calibrate deformable mirror
        """
        # Create calibration worker and thread
        calib_thread = QThread()
        calib_thread.setObjectName('calib_thread')
        calib_worker = Calibration(mirror)
        calib_worker.moveToThread(calib_thread)

        # Connect to signals
        calib_thread.started.connect(calib_worker.run)
        calib_worker.image.connect(lambda obj: self.handle_image_disp(obj))
        calib_worker.info.connect(lambda obj: self.handle_mirror_info(obj))
        calib_worker.error.connect(lambda obj: self.handle_error(obj))
        calib_worker.done.connect(self.handle_calib_done)

        # Store calibration worker and thread
        self.workers['calib_worker'] = calib_worker
        self.threads['calib_thread'] = calib_thread

        # Start calibration thread
        calib_thread.start()

    #========== Signal handlers ==========#
    def handle_layer_disp(self, obj):
        """
        Handle display of search block layer
        """
        self.main.update_image(obj, flag = 0)

    def handle_image_disp(self, obj):
        """
        Handle display of S-H spot images
        """
        self.main.update_image(obj, flag = 1)

    def handle_SB_info(self, obj):
        """
        Handle search block geometry and centroiding information
        """
        self.SB_info.update(obj)

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

    def handle_cent_start(self):
        """
        Handle start of S-H spot centroid calculation
        """
        self.get_centroids(self.devices['sensor'], self.SB_info)

    def handle_cent_done(self):
        """
        Handle end of S-H spot centroid calculation
        """
        self.threads['cent_thread'].quit()
        self.threads['cent_thread'].wait()
        self.main.ui.centroidBtn.setChecked(False)

    def handle_calib_start(self):
        """
        Handle start of deformable mirror calibration
        """
        self.calibrate_DM(self.devices['sensor'], self.devices['mirror'], self.SB_info)

    def handle_mirror_info(self, obj):
        """
        Handle deformable mirror information
        """
        self.mirror_info.update(obj)

    def handle_calib_done(self):
        """
        Handle end of deformable mirror calibration
        """
        self.threads['calib_thread'].quit()
        self.threads['calib_thread'].wait()
        self.main.ui.calibrateBtn.setChecked(False)


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