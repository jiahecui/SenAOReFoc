from PySide2.QtWidgets import QApplication, QStyleFactory, QMainWindow, QFileDialog, QDialog, QVBoxLayout, QGridLayout, QWidget, QSpacerItem, QSizePolicy
from PySide2.QtCore import QThread, QObject, Slot, Signal, QSize, QTimer

import qtawesome as qta

import os
import numpy as np

from sensorbasedAO.gui.ui.main import Ui_MainWindow
from sensorbasedAO.gui.common import FloatingWidget
from sensorbasedAO.config import config

class Main(QMainWindow):
    """
    The main GUI window
    """
    def __init__(self, app, parent = None, debug = False):
        super().__init__(parent)

        self.debug = debug
        self.app = app

        # Setup and initialise GUI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Initialise instance variables
        self.prev_settings = {}

        # Main handlers
        self.ui.initialiseBtn.clicked.connect(self.on_initialise)
        self.ui.centroidBtn.clicked.connect(self.on_centroid)
        self.ui.calibrateBtn.clicked.connect(self.on_calibrate)
        # self.ui.stopBtn.clicked.connect(self.on_stop)
        self.ui.quitBtn.clicked.connect(self.on_quit)

    def update_image(self, image, flag):
        """
        Update image on S-H viewer

        Args: 
            image as numpy array
            flag = 0 for search block layer display
            flag = 1 for S-H spot image display
        """
        self.ui.SHViewer.set_image(image, flag)       
        self.image_temp = image

    #========== GUI event handlers ==========#
    def on_initialise(self, checked):
        """
        Search block initialisation button handler
        """
        btn = self.sender()

        # Initialise search blocks if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.setup_SB()
            btn.setChecked(True)

    def on_centroid(self, checked):
        """
        S-H spot centroid calculation handler
        """
        btn = self.sender()

        # Start calculating S-H spot centroids if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_cent_start()
            btn.setChecked(True)

    def on_calibrate(self, checked):
        """
        Deformable mirror calibration handler
        """
        btn = self.sender()

        # Start calibrating deformable mirror if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_calib_start()
            btn.setChecked(True)
            
    def on_quit(self):
        self.close()


        