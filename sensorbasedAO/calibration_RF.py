from PySide2.QtCore import QThread, QObject, Signal, Slot
from PySide2.QtWidgets import QApplication

import sys
import os
import argparse
import time
import click
import h5py
import numpy as np

import log
from config import config
from image_acquisition import acq_image
from centroid_acquisition import acq_centroid

logger = log.get_logger(__name__)

class Calibration_RF(QObject):
    """
    Calibrates remote focusing using nulling correction 
    """
    start = Signal()
    write = Signal()
    done = Signal()
    error = Signal(object)
    image = Signal(object)
    message = Signal(object)
    info = Signal(object)

    def __init__(self, sensor, mirror, settings):

        # Get search block settings
        self.SB_settings = settings['SB_info']

        # Get mirror settings
        self.mirror_settings = settings['mirror_info']

        # Get AO settings
        self.AO_settings = settings['AO_info']

        # Get sensor instance
        self.sensor = sensor

        # Get mirror instance
        self.mirror = mirror

        # Initialise Zernike coefficient array
        self.zern_coeff = np.zeros(config['AO']['control_coeff_num'])

        # Initialise array to store remote focusing calibration voltages
        self.calib_array = np.zeros([config['AO']['control_coeff_num'], config['RF_calib']['step_num'] * 2 + 1])

        # Choose working DM along with its parameters
        if config['DM']['DM_num'] == 0:
            self.actuator_num = config['DM0']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_0']
        elif config['DM']['DM_num'] == 1:
            self.actuator_num = config['DM1']['actuator_num']
            self.pupil_diam = config['search_block']['pupil_diam_1']

        super().__init__()

    @Slot(object)
    def run(self):
        try:
            # Set process flags
            self.loop = True

            # Start thread
            self.start.emit()

            self.message.emit('\nProcess started for calibration of remote focusing...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(self.actuator_num)

            # Iterate through each step and retrieve the voltages that correct for the aberration at that step
            # for i in range(config['RF_calib']['step_num']):

        except Exception as e:
            raise
            self.error.emit(e)

