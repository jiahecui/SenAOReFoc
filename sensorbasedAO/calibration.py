from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import logging
import sys
import os
import argparse
import time
import PIL.Image
import numpy as np

import log
from config import config
from image_acquisition import acq_image

logger = log.get_logger(__name__)

class Calibration(QObject):
    """
    Calibrates deformable mirror and retrieves influence function
    """
    start = Signal()
    done = Signal()
    error = Signal(object)
    image = Signal(object)
    layer = Signal(object)
    info = Signal(object)

    def __init__(self, sensor, mirror, settings):

        # Get search block settings
        self.SB_settings = settings

        # Get sensor instance
        self.sensor = sensor

        # Get mirror instance
        self.mirror = mirror

        # Initialise influence function matrix
        self.inf_matrix = np.zeros([2 * self.SB_settings['act_ref_cent_num'], config['DM']['actuator_num']])


    # @Slot(object)
    # def run(self):
    #     try:
    #         self.start.emit()

    #         """
    #         Apply highest and lowest voltage to each actuator individually and retrieve raw slopes of each S-H spot correspondingly
    #         """
    #         for i in range(config['DM']['actuator_num']):
                