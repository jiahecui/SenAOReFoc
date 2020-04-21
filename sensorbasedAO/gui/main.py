from PySide2.QtWidgets import QApplication, QStyleFactory, QMainWindow, QFileDialog, QDialog, QVBoxLayout, QGridLayout, QWidget, QSpacerItem, QSizePolicy
from PySide2.QtCore import QThread, QObject, Slot, Signal, QSize, QTimer

import qtawesome as qta
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
        self.ui.positionBtn.clicked.connect(self.on_position)
        self.ui.centroidBtn.clicked.connect(self.on_centroid)
        self.ui.calibrateBtn.clicked.connect(self.on_calibrate)
        self.ui.conversionBtn.clicked.connect(self.on_conversion)
        self.ui.calibrateBtn_2.clicked.connect(self.on_calibrate_zern)
        self.ui.ZernikeOKBtn.clicked.connect(self.get_zernike_settings)
        self.ui.ZernikeTestBtn.clicked.connect(self.on_zern_test)
        self.ui.ZernikeAOBtn_1.clicked.connect(self.on_zern_AO_1)
        self.ui.ZernikeAOBtn_2.clicked.connect(self.on_zern_AO_2)
        self.ui.ZernikeAOBtn_3.clicked.connect(self.on_zern_AO_3)
        self.ui.slopeAOBtn_1.clicked.connect(self.on_slope_AO_1)
        self.ui.slopeAOBtn_2.clicked.connect(self.on_slope_AO_2)
        self.ui.slopeAOBtn_3.clicked.connect(self.on_slope_AO_3)
        self.ui.ZernikeFullBtn.clicked.connect(self.on_zern_AO_full)
        self.ui.SlopeFullBtn.clicked.connect(self.on_slope_AO_full)
        self.ui.stopBtn.clicked.connect(self.on_stop)
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

    def update_message(self, text):
        """
        Update message info in display box
        """
        self.ui.displayBox.appendPlainText(text)

    #========== GUI event handlers ==========#
    def on_initialise(self, checked):
        """
        Search block initialisation handler
        """
        btn = self.sender()

        # Initialise search blocks if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.setup_SB()
            btn.setChecked(True)

    def on_position(self, checked):
        """
        Search block positioning handler
        """
        btn = self.sender()

        # Start positioning search blocks if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_pos_start()
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
        Deformable mirror calibration handler via slopes
        """
        btn = self.sender()

        # Start calibrating deformable mirror if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_calib_start()
            btn.setChecked(True)

    def on_conversion(self, checked):
        """
        Slope - Zernike conversion matrix generating button handler
        """
        btn = self.sender()

        # Generate slope - zernike conversion matrix if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_conv_matrix()
            btn.setChecked(True)

    def on_calibrate_zern(self, checked):
        """
        Deformable mirror calibration handler via zernikes
        """
        btn = self.sender()

        # Generate control matrix via Zernikes if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_calib2_start()
            btn.setChecked(True)

    def get_zernike_settings(self, checked):
        """
        Zernike coefficient array value update handler
        """
        btn = self.sender()

        # Get Zernike coefficient array values if pressed 
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            if not self.ui.ZernikeArrEdt.text() is '':
                try:
                    zernike_array = self.ui.ZernikeArrEdt.text()
                    zernike_array = [float(i) for i in zernike_array.split(' ')]
                except Exception as e:
                    btn.setChecked(False)
                    self.app.handle_message_disp('Invalid input. Please try again.')
            elif not self.ui.ZernikeValSpin.value() == 0 or self.ui.ZernikeCoeffSpin.value() == 0:
                try:
                    zernike_coeff = self.ui.ZernikeCoeffSpin.value()
                    zernike_array = np.zeros(zernike_coeff)
                    zernike_array[-1] = self.ui.ZernikeValSpin.value()
                except Exception as e:
                    btn.setChecked(False)
                    self.app.handle_message_disp('Invalid input. Please try again.')
            else:
                btn.setChecked(False)
                self.app.handle_message_disp('Please enter zernike coefficients.')

            if btn.isChecked():
                if not len(zernike_array) > config['AO']['recon_coeff_num']:
                    settings = {}
                    settings['zernike_array_test'] = zernike_array
                    self.app.handle_SB_info(settings)
                    self.app.write_SB_info()
                    self.app.handle_message_disp('Zernike coefficients loaded.')
                    btn.setChecked(False)
                else:
                    self.app.handle_message_disp('Input too long. Please try again.')
                    btn.setChecked(False)       

    def on_zern_test(self, checked):
        """
        Closed-loop AO control via Zernikes test 1 handler
        """
        btn = self.sender()

        # Test closed-loop AO control via Zernikes if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_zern_test_start()
            btn.setChecked(True)

    def on_zern_AO_1(self, checked):
        """
        Closed-loop AO control via Zernikes handler 1
        """
        btn = self.sender()

        # Start closed-loop AO control via Zernikes if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_zern_AO_start(mode = 1)
            btn.setChecked(True)

    def on_zern_AO_2(self, checked):
        """
        Closed-loop AO control via Zernikes handler 2
        """
        btn = self.sender()

        # Start closed-loop AO control via Zernikes if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_zern_AO_start(mode = 2)
            btn.setChecked(True)

    def on_zern_AO_3(self, checked):
        """
        Closed-loop AO control via Zernikes handler 3
        """
        btn = self.sender()

        # Start closed-loop AO control via Zernikes if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_zern_AO_start(mode = 3)
            btn.setChecked(True)

    def on_slope_AO_1(self, checked):
        """
        Closed-loop AO control via slopes handler 1
        """
        btn = self.sender()

        # Start closed-loop AO control via slopes if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_slope_AO_start(mode = 1)
            btn.setChecked(True)

    def on_slope_AO_2(self, checked):
        """
        Closed-loop AO control via slopes handler 2
        """
        btn = self.sender()

        # Start closed-loop AO control via slopes if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_slope_AO_start(mode = 2)
            btn.setChecked(True)

    def on_slope_AO_3(self, checked):
        """
        Closed-loop AO control via slopes handler 3
        """
        btn = self.sender()

        # Start closed-loop AO control via slopes if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_slope_AO_start(mode = 3)
            btn.setChecked(True)

    def on_zern_AO_full(self, checked):
        """
        Full closed-loop AO control via Zernikes handler 
        """
        btn = self.sender()

        # Start full closed-loop AO control via Zernikes if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_zern_AO_start(mode = 4)
            btn.setChecked(True)

    def on_slope_AO_full(self, checked):
        """
        Full closed-loop AO control via slopes handler
        """
        btn = self.sender()

        # Start full closed-loop AO control via slopes if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            self.app.handle_slope_AO_start(mode = 4)
            btn.setChecked(True)

    def on_stop(self):
        """
        Stop threads and devices
        """
        self.app.stop()
            
    def on_quit(self):
        """
        Quit application
        """
        self.app.quit()


        