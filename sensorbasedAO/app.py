from PySide2.QtWidgets import QApplication, QStyleFactory, QMainWindow, QFileDialog, QDialog, QVBoxLayout, QWidget
from PySide2.QtCore import QThread, QObject, Slot, Signal, QSize, QTimer

import os
os.environ['QT_API'] = 'pyside2'

import qtawesome as qta

import numpy as np
from qimage2ndarray import array2qimage, gray2qimage
import imageio

import threading
from queue import Queue
import argparse
from functools import partial
import time
from datetime import datetime
from pathlib import Path
import logging
import sys

import sensorbasedAO
import sensorbasedAO.log
from sensorbasedAO.fpga import FPGA

from ximea import xiapi
from alpao import asdk

logger = doptical.log.get_logger()


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