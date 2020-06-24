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
        self.ui.scanFocusCheck.stateChanged.connect(self.on_focussettings)
        self.ui.focusDepthSpin.valueChanged.connect(self.on_focussettings)
        self.ui.stepIncreSpin.valueChanged.connect(self.on_focussettings)
        self.ui.stepNumSpin.valueChanged.connect(self.on_focussettings)
        self.ui.startDepthSpin.valueChanged.connect(self.on_focussettings)
        self.ui.pauseTimeSpin.valueChanged.connect(self.on_focussettings)
        self.ui.AOTypeCombo.currentIndexChanged.connect(self.on_focussettings)
        self.ui.moveBtn.clicked.connect(self.on_move)
        self.ui.scanBtn.clicked.connect(self.on_scan)
        self.ui.MLDataBtn.clicked.connect(self.on_ML_dataset)
        self.ui.stopBtn.clicked.connect(self.on_stop)
        self.ui.quitBtn.clicked.connect(self.on_quit)

        # One time initialisation of remote focusing settings
        self.on_focussettings()

    #=============== Methods ===============#
    def get_focus_settings(self):
        """
        Retrieve remote focusing settings from GUI

        Returns:
            dict: dict of settings
        """
        # Retrieve GUI remote focusing depth scan settings
        focus_depth = self.ui.focusDepthSpin.value()
        step_incre = self.ui.stepIncreSpin.value()
        step_num = self.ui.stepNumSpin.value()
        start_depth = self.ui.startDepthSpin.value()
        pause_time = self.ui.pauseTimeSpin.value()

        # Convert depth parameters to amounts of defocus to apply on DM
        focus_depth_defoc = focus_depth / 2
        step_incre_defoc = step_incre / 2
        start_depth_defoc = start_depth / 2

        # Retrieve GUI remote focusing AO correction settings
        AO_type = self.ui.AOTypeCombo.currentIndex()

        AO_types = {
            0: 'None',
            1: 'Zernike AO 3',
            2: 'Zernike Full',
            3: 'Slope AO 3',
            4: 'Slope Full'
        }

        # Create remote focusing settings dict
        focus_settings = {}
        focus_settings['focus_depth_defoc'] = focus_depth_defoc
        focus_settings['step_incre_defoc'] = step_incre_defoc
        focus_settings['step_num'] = step_num
        focus_settings['start_depth_defoc'] = start_depth_defoc
        focus_settings['pause_time'] = pause_time
        focus_settings['AO_type'] = AO_type

        return focus_settings

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
                    self.app.handle_message_disp('\nInvalid input. Please try again.')
            elif not self.ui.ZernikeValSpin.value() == 0 or self.ui.ZernikeCoeffSpin.value() == 0:
                try:
                    zernike_coeff = self.ui.ZernikeCoeffSpin.value()
                    zernike_array = np.zeros(zernike_coeff)
                    zernike_array[-1] = self.ui.ZernikeValSpin.value()
                except Exception as e:
                    btn.setChecked(False)
                    self.app.handle_message_disp('\nInvalid input. Please try again.')
            else:
                btn.setChecked(False)
                self.app.handle_message_disp('\nPlease enter zernike coefficients.')

            if btn.isChecked():
                if not len(zernike_array) > config['AO']['recon_coeff_num']:
                    settings = {}
                    settings['zernike_array_test'] = zernike_array
                    self.app.handle_SB_info(settings)
                    self.app.write_SB_info()
                    self.app.handle_message_disp('\nZernike coefficients loaded.')
                    btn.setChecked(False)
                else:
                    self.app.handle_message_disp('\nInput too long. Please try again.')
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
            # Start zern_test
            self.app.handle_zern_test_start(mode = config['zern_test']['zern_test_mode'])
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
            # Set remote focusing flag to 0 and update AO_info
            remote_focusing = {}
            remote_focusing['focus_enable'] = 0
            self.app.handle_AO_info(remote_focusing)
            self.app.write_AO_info()
            
            # Start zern_AO_1
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
            # Set remote focusing flag to 0 and update AO_info
            remote_focusing = {}
            remote_focusing['focus_enable'] = 0
            self.app.handle_AO_info(remote_focusing)
            self.app.write_AO_info()
            
            # Start zern_AO_2
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
            # Set remote focusing flag to 0 and update AO_info
            remote_focusing = {}
            remote_focusing['focus_enable'] = 0
            self.app.handle_AO_info(remote_focusing)
            self.app.write_AO_info()
            
            # Start zern_AO_3
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
            # Set remote focusing flag to 0 and update AO_info
            remote_focusing = {}
            remote_focusing['focus_enable'] = 0
            self.app.handle_AO_info(remote_focusing)
            self.app.write_AO_info()
            
            # Start slope_AO_1
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
            # Set remote focusing flag to 0 and update AO_info
            remote_focusing = {}
            remote_focusing['focus_enable'] = 0
            self.app.handle_AO_info(remote_focusing)
            self.app.write_AO_info()
            
            # Start slope_AO_2
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
            # Set remote focusing flag to 0 and update AO_info
            remote_focusing = {}
            remote_focusing['focus_enable'] = 0
            self.app.handle_AO_info(remote_focusing)
            self.app.write_AO_info()
            
            # Start slope_AO_3
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
            # Set remote focusing flag to 0 and update AO_info
            remote_focusing = {}
            remote_focusing['focus_enable'] = 0
            self.app.handle_AO_info(remote_focusing)
            self.app.write_AO_info()
            
            # Start zern_AO_full
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
            # Set remote focusing flag to 0 and update AO_info
            remote_focusing = {}
            remote_focusing['focus_enable'] = 0
            self.app.handle_AO_info(remote_focusing)
            self.app.write_AO_info()
            
            # Start slope_AO_full
            self.app.handle_slope_AO_start(mode = 4)
            btn.setChecked(True)

    def on_focussettings(self):
        """
        Remote focusing settings update handler
        """
        # Retrieve remote focusing settings from GUI
        settings = self.get_focus_settings()

        # Update GUI
        if self.ui.scanFocusCheck.isChecked():
            self.ui.focusDepthSpin.setReadOnly(True)
            self.ui.stepIncreSpin.setReadOnly(False)
            self.ui.stepNumSpin.setReadOnly(False)
            self.ui.startDepthSpin.setReadOnly(False)
            self.ui.pauseTimeSpin.setReadOnly(False)
        else:
            self.ui.focusDepthSpin.setReadOnly(False)
            self.ui.stepIncreSpin.setReadOnly(True)
            self.ui.stepNumSpin.setReadOnly(True)
            self.ui.startDepthSpin.setReadOnly(True)
            self.ui.pauseTimeSpin.setReadOnly(True)

    def on_move(self, checked):
        """
        Focus move handler
        """
        btn = self.sender()

        # Stop running focus scan
        self.app.handle_focus_done(1)
        self.ui.scanBtn.setChecked(False)

        # Move focus if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            # Retrieve remote focusing settings from GUI
            settings = self.get_focus_settings()
            settings['step_num'] = 1
            settings['focus_mode_flag'] = 0

            # Set remote focusing flag to 1 and update AO_info
            remote_focusing = {}
            remote_focusing['focus_enable'] = 1
            self.app.handle_AO_info(remote_focusing)
            self.app.write_AO_info()
            
            # Start focus move
            self.app.handle_focusing_info(settings)
            self.app.write_focusing_info()
            self.app.handle_focus_start(AO_type = settings['AO_type'])
            btn.setChecked(True)

    def on_scan(self, checked):
        """
        Focus scan start handler
        """
        btn = self.sender()

        # Stop running focus move
        self.app.handle_focus_done(0)
        self.ui.moveBtn.setChecked(False)

        # Scan focus if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            # Retrieve remote focusing settings from GUI
            settings = self.get_focus_settings()
            settings['focus_mode_flag'] = 1

            # Set remote focusing flag to 1 and update AO_info
            remote_focusing = {}
            remote_focusing['focus_enable'] = 1
            self.app.handle_AO_info(remote_focusing)
            self.app.write_AO_info()

            # Start focus move
            self.app.handle_focusing_info(settings)
            self.app.write_focusing_info()
            self.app.handle_focus_start(AO_type = settings['AO_type'])
            btn.setChecked(True)

    def on_ML_dataset(self, checked):
        """
        Generate ML dataset handler
        """
        btn = self.sender()

        # Generate ML dataset if pressed
        if not btn.isChecked():
            btn.setChecked(False)
        else:
            # Start ML_dataset
            self.app.handle_ML_dataset_start()
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


        