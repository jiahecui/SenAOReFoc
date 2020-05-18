from PySide2.QtWidgets import QApplication, QStyleFactory, QMainWindow, QFileDialog, QDialog, QVBoxLayout, QWidget
from PySide2.QtCore import QThread, QObject, Slot, Signal, QSize, QTimer

import logging
import sys
import os
import argparse
import time
import h5py
import numpy as np

from datetime import datetime

from alpao.Lib import asdk  # Use alpao.Lib for 32-bit applications and alpao.Lib64 for 64-bit applications
from ximea import xiapi

import log
from config import config
from sensor import SENSOR
from mirror import MIRROR
from gui.main import Main
from SB_geometry import Setup_SB
from SB_position import Positioning
from centroiding import Centroiding
from calibration import Calibration
from conversion import Conversion
from calibration_zern import Calibration_Zern
from AO_zernikes_test import AO_Zernikes_Test
from AO_zernikes import AO_Zernikes
from AO_slopes import AO_Slopes

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

        # Initialise dictionary for storing data info throughout processing of software
        self.data_info = {'SB_info': {}, 'mirror_info': {}, 'centroiding_info': {}, 'AO_info': {}, 'centroiding_img': {}, \
            'calibration_img': {}, 'AO_img': {}, 'focusing_info': {}}
        self.AO_group = {'zern_test': {}, 'zern_AO_1': {}, 'zern_AO_2': {}, 'zern_AO_3': {}, 'zern_AO_full': {}, \
            'slope_AO_1': {}, 'slope_AO_2': {}, 'slope_AO_3': {}, 'slope_AO_full': {}}

        # Initialise output HDF5 file
        self.HDF5_init()

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
        if config['dummy']:
            sensor = SENSOR.get('debug')
        else:
            try:
                sensor = SENSOR.get(config['camera']['SN'])
                sensor.open_device_by_SN(config['camera']['SN'])
                print('Sensor load success.')
            except Exception as e:
                logger.warning('Sensor load error', e)
                sensor = None

        self.devices['sensor'] = sensor
        
        # Add deformable mirror
        try:
            mirror = MIRROR.get(config['DM']['SN'])
            print('Mirror load success.')
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
        SB_worker.message.connect(lambda obj: self.handle_message_disp(obj))
        SB_worker.info.connect(lambda obj: self.handle_SB_info(obj))
        SB_worker.error.connect(lambda obj: self.handle_error(obj))
        SB_worker.done.connect(self.handle_SB_done)

        # Store SB worker and thread
        self.workers['SB_worker'] = SB_worker
        self.threads['SB_thread'] = SB_thread

        # Start SB thread
        SB_thread.start()

    def position_SB(self, sensor, SB_info):
        """
        Position search blocks to appropriate position
        """
        # Create positioning worker and thread
        pos_thread = QThread()
        pos_thread.setObjectName('pos_thread')
        pos_worker = Positioning(sensor, SB_info)
        pos_worker.moveToThread(pos_thread)

        # Connect to signals
        pos_thread.started.connect(pos_worker.run)
        pos_worker.layer.connect(lambda obj: self.handle_layer_disp(obj))
        pos_worker.image.connect(lambda obj: self.handle_image_disp(obj))
        pos_worker.message.connect(lambda obj: self.handle_message_disp(obj))
        pos_worker.SB_info.connect(lambda obj: self.handle_SB_info(obj))
        pos_worker.mirror_info.connect(lambda obj: self.handle_mirror_info(obj))
        pos_worker.write.connect(self.write_SB_info)
        pos_worker.error.connect(lambda obj: self.handle_error(obj))
        pos_worker.done.connect(self.handle_pos_done)

        # Store positioning worker and thread
        self.workers['pos_worker'] = pos_worker
        self.threads['pos_thread'] = pos_thread

        # Start positioning thread
        pos_thread.start()

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
        cent_worker.message.connect(lambda obj: self.handle_message_disp(obj))
        cent_worker.info.connect(lambda obj: self.handle_centroiding_info(obj))
        cent_worker.write.connect(self.write_centroiding_info)
        cent_worker.error.connect(lambda obj: self.handle_error(obj))
        cent_worker.done.connect(self.handle_cent_done)

        # Store centroiding worker and thread
        self.workers['cent_worker'] = cent_worker
        self.threads['cent_thread'] = cent_thread

        # Start centroiding thread
        cent_thread.start()

    def calibrate_DM(self, sensor, mirror, SB_info):
        """
        Calibrate deformable mirror and get control matrix via slopes
        """
        # Create calibration worker and thread
        calib_thread = QThread()
        calib_thread.setObjectName('calib_thread')
        calib_worker = Calibration(sensor, mirror, SB_info)
        calib_worker.moveToThread(calib_thread)

        # Connect to signals
        calib_thread.started.connect(calib_worker.run)
        calib_worker.image.connect(lambda obj: self.handle_image_disp(obj))
        calib_worker.message.connect(lambda obj: self.handle_message_disp(obj))
        calib_worker.info.connect(lambda obj: self.handle_mirror_info(obj))
        calib_worker.write.connect(self.write_mirror_info)
        calib_worker.error.connect(lambda obj: self.handle_error(obj))
        calib_worker.done.connect(self.handle_calib_done)

        # Store calibration worker and thread
        self.workers['calib_worker'] = calib_worker
        self.threads['calib_thread'] = calib_thread

        # Start calibration thread
        calib_thread.start()

    def get_conv_matrix(self, SB_info):
        """
        Generate slope - zernike conversion matrix
        """
        # Create conversion worker and thread
        conv_thread = QThread()
        conv_thread.setObjectName('conv_thread')
        conv_worker = Conversion(SB_info)
        conv_worker.moveToThread(conv_thread)

        # Connect to signals
        conv_thread.started.connect(conv_worker.run)
        conv_worker.message.connect(lambda obj: self.handle_message_disp(obj))
        conv_worker.info.connect(lambda obj: self.handle_mirror_info(obj))
        conv_worker.write.connect(self.write_mirror_info)
        conv_worker.error.connect(lambda obj: self.handle_error(obj))
        conv_worker.done.connect(self.handle_conv_done)

        # Store conversion worker and thread
        self.workers['conv_worker'] = conv_worker
        self.threads['conv_thread'] = conv_thread

        # Start conversion thread
        conv_thread.start()

    def calibrate_DM_zern(self, data_info):
        """
        Get deformable mirror control matrix via zernikes
        """
        # Create calibration worker and thread
        calib2_thread = QThread()
        calib2_thread.setObjectName('calib2_thread')
        calib2_worker = Calibration_Zern(data_info)
        calib2_worker.moveToThread(calib2_thread)

        # Connect to signals
        calib2_thread.started.connect(calib2_worker.run)
        calib2_worker.message.connect(lambda obj: self.handle_message_disp(obj))
        calib2_worker.info.connect(lambda obj: self.handle_mirror_info(obj))
        calib2_worker.write.connect(self.write_mirror_info)
        calib2_worker.error.connect(lambda obj: self.handle_error(obj))
        calib2_worker.done.connect(self.handle_calib2_done)

        # Store calibration worker and thread
        self.workers['calib2_worker'] = calib2_worker
        self.threads['calib2_thread'] = calib2_thread

        # Start calibration thread
        calib2_thread.start()

    def control_zern_test(self, sensor, mirror, data_info, mode):
        """
        Closed-loop AO control via Zernikes test run
        """
        # Create Zernike AO worker and thread
        zern_thread = QThread()
        zern_thread.setObjectName('zern_thread')
        zern_worker = AO_Zernikes_Test(sensor, mirror, data_info)
        zern_worker.moveToThread(zern_thread)

        # Connect to signals
        if mode == 0:
            zern_thread.started.connect(zern_worker.run0)
        elif mode == 1:
            zern_thread.started.connect(zern_worker.run1)

        zern_worker.done.connect(self.handle_zern_test_done)
        zern_worker.image.connect(lambda obj: self.handle_image_disp(obj))   
        zern_worker.message.connect(lambda obj: self.handle_message_disp(obj))
        zern_worker.info.connect(lambda obj: self.handle_AO_info(obj))
        zern_worker.write.connect(self.write_AO_info)
        zern_worker.error.connect(lambda obj: self.handle_error(obj))

        # Store Zernike AO worker and thread
        self.workers['zern_worker'] = zern_worker
        self.threads['zern_thread'] = zern_thread

        # Start Zernike AO thread
        zern_thread.start()

    def control_zern_AO(self, sensor, mirror, data_info, mode):
        """
        Closed-loop AO control via Zernikes + remote focusing
        """
        # Create Zernike AO worker and thread
        zern_AO_thread = QThread()
        zern_AO_thread.setObjectName('zern_AO_thread')
        zern_AO_worker = AO_Zernikes(sensor, mirror, data_info)
        zern_AO_worker.moveToThread(zern_AO_thread)

        # Connect to signals
        if mode == 1:
            zern_AO_thread.started.connect(zern_AO_worker.run1)
        elif mode == 2:
            zern_AO_thread.started.connect(zern_AO_worker.run2)          
        elif mode == 3:
            zern_AO_thread.started.connect(zern_AO_worker.run3)            
        elif mode == 4:
            zern_AO_thread.started.connect(zern_AO_worker.run4)
        elif mode == 5:
            zern_AO_thread.started.connect(zern_AO_worker.run5)

        zern_AO_worker.done.connect(lambda mode: self.handle_zern_AO_done(mode))
        zern_AO_worker.done2.connect(lambda mode: self.handle_focus_done(mode))
        zern_AO_worker.image.connect(lambda obj: self.handle_image_disp(obj))   
        zern_AO_worker.message.connect(lambda obj: self.handle_message_disp(obj))
        zern_AO_worker.info.connect(lambda obj: self.handle_AO_info(obj))
        zern_AO_worker.write.connect(self.write_AO_info)
        zern_AO_worker.error.connect(lambda obj: self.handle_error(obj))       

        # Store Zernike AO worker and thread
        self.workers['zern_AO_worker'] = zern_AO_worker
        self.threads['zern_AO_thread'] = zern_AO_thread

        # Start Zernike AO thread
        zern_AO_thread.start()

    def control_slope_AO(self, sensor, mirror, data_info, mode):
        """
        Closed-loop AO control via slopes
        """
        # Create slopes AO worker and thread
        slopes_AO_thread = QThread()
        slopes_AO_thread.setObjectName('slopes_AO_thread')
        slopes_AO_worker = AO_Slopes(sensor, mirror, data_info)
        slopes_AO_worker.moveToThread(slopes_AO_thread)

        # Connect to signals
        if mode == 1:
            slopes_AO_thread.started.connect(slopes_AO_worker.run1)
        elif mode == 2:
            slopes_AO_thread.started.connect(slopes_AO_worker.run2)
        elif mode == 3:
            slopes_AO_thread.started.connect(slopes_AO_worker.run3)
        elif mode == 4:
            slopes_AO_thread.started.connect(slopes_AO_worker.run4)
            
        slopes_AO_worker.done.connect(lambda mode: self.handle_slope_AO_done(mode))
        slopes_AO_worker.done2.connect(lambda mode: self.handle_focus_done(mode))
        slopes_AO_worker.image.connect(lambda obj: self.handle_image_disp(obj))   
        slopes_AO_worker.message.connect(lambda obj: self.handle_message_disp(obj))
        slopes_AO_worker.info.connect(lambda obj: self.handle_AO_info(obj))
        slopes_AO_worker.write.connect(self.write_AO_info)
        slopes_AO_worker.error.connect(lambda obj: self.handle_error(obj))       

        # Store slopes AO worker and thread
        self.workers['slopes_AO_worker'] = slopes_AO_worker
        self.threads['slopes_AO_thread'] = slopes_AO_thread

        # Start slopes AO thread
        slopes_AO_thread.start()

    #========== Signal handlers ==========#
    def HDF5_init(self):
        """
        Initialises HDF5 file and creates relevant groups
        """
        self.output_data = h5py.File('data_info.h5', 'a')
        data_keys = list(self.data_info.keys())
        AO_keys = list(self.AO_group.keys())
        for k in data_keys:
            if not k in self.output_data:
                self.output_data.create_group(k)
        for k in AO_keys:
            if not k in self.output_data['AO_img']:
                self.output_data['AO_img'].create_group(k)
            if not k in self.output_data['AO_info']:
                self.output_data['AO_info'].create_group(k)      
        self.output_data.close()

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

    def handle_message_disp(self, obj):
        """"
        Handle display of message info
        """
        self.main.update_message(obj)

    def handle_SB_info(self, obj):
        """
        Handle search block geometry and centroiding information
        """
        self.data_info['SB_info'].update(obj)

    def write_SB_info(self):
        """
        Write search block info to HDF5 file
        """
        self.output_data = h5py.File('data_info.h5', 'a')
        grp1 = self.output_data['SB_info']
        for k, v in self.data_info['SB_info'].items():
            if k in grp1:
                del grp1[k]
            grp1.create_dataset(k, data = v)           
        self.output_data.close()
    
    def handle_mirror_info(self, obj):
        """
        Handle deformable mirror information
        """
        self.data_info['mirror_info'].update(obj)

    def write_mirror_info(self):
        """
        Write mirror info to HDF5 file
        """
        self.output_data = h5py.File('data_info.h5', 'a')
        grp2 = self.output_data['mirror_info']
        for k, v in self.data_info['mirror_info'].items():
            if k in grp2:
               del grp2[k]
            grp2.create_dataset(k, data = v)
        self.output_data.close()

    def handle_centroiding_info(self, obj):
        """
        Handle centroiding information
        """
        self.data_info['centroiding_info'].update(obj)

    def write_centroiding_info(self):
        """
        Write centroiding info to HDF5 file
        """
        self.output_data = h5py.File('data_info.h5', 'a')
        grp3 = self.output_data['centroiding_info']
        for k, v in self.data_info['centroiding_info'].items():
            if k in grp3:
               del grp3[k]
            grp3.create_dataset(k, data = v)
        self.output_data.close()

    def handle_AO_info(self, obj):
        """
        Handle AO information
        """
        self.data_info['AO_info'].update(obj)
        
    def write_AO_info(self):
        """
        Write AO info to HDF5 file
        """
        self.output_data = h5py.File('data_info.h5', 'a')
        grp4 = self.output_data['AO_info']
        for k, v in self.data_info['AO_info'].items():
            if k in grp4:
                if isinstance(v, dict):  
                    for kk, vv in self.data_info['AO_info'][k].items():
                        if kk in grp4[k]:
                            del grp4[k][kk]
                        grp4[k].create_dataset(kk, data = vv) 
                else:
                    del grp4[k]
                    grp4.create_dataset(k, data = v)
            else:
                if isinstance(v, dict):  
                    for kk, vv in self.data_info['AO_info'][k].items():
                        grp4[k].create_dataset(kk, data = vv)  
                else:
                    grp4.create_dataset(k, data = v)            
        self.output_data.close()

    def handle_focusing_info(self, obj):
        """
        Handle remote focusing information
        """
        self.data_info['focusing_info'].update(obj)

    def write_focusing_info(self):
        """
        Write focusing info into HDF5 file
        """
        self.output_data = h5py.File('data_info.h5', 'a')
        grp5 = self.output_data['focusing_info']
        for k, v in self.data_info['focusing_info'].items():
            if k in grp5:
               del grp5[k]
            grp5.create_dataset(k, data = v)
        self.output_data.close()

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

    def handle_pos_start(self):
        """
        Handle start of search block posiioning
        """
        self.position_SB(self.devices['sensor'], self.data_info['SB_info'])

    def handle_pos_done(self):
        """
        Handle end of search block posiioning
        """
        self.threads['pos_thread'].quit()
        self.threads['pos_thread'].wait()
        self.main.ui.positionBtn.setChecked(False)

    def handle_cent_start(self):
        """
        Handle start of S-H spot centroid calculation
        """
        self.get_centroids(self.devices['sensor'], self.data_info['SB_info'])

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
        self.calibrate_DM(self.devices['sensor'], self.devices['mirror'], self.data_info['SB_info'])

    def handle_calib_done(self):
        """
        Handle end of deformable mirror calibration
        """
        self.threads['calib_thread'].quit()
        self.threads['calib_thread'].wait()
        self.main.ui.calibrateBtn.setChecked(False)

    def handle_conv_matrix(self):
        """
        Handle slope - zernike conversion matrix generation
        """
        self.get_conv_matrix(self.data_info['SB_info'])

    def handle_conv_done(self):
        """
        Handle end of slope - zernike conversion matrix generation
        """
        self.threads['conv_thread'].quit()
        self.threads['conv_thread'].wait()
        self.main.ui.conversionBtn.setChecked(False)

    def handle_calib2_start(self):
        """
        Handle start of deformable mirror calibration via zernikes
        """
        self.calibrate_DM_zern(self.data_info)

    def handle_calib2_done(self):
        """
        Handle end of deformable mirror calibration via zernikes
        """
        self.threads['calib2_thread'].quit()
        self.threads['calib2_thread'].wait()
        self.main.ui.calibrateBtn_2.setChecked(False)

    def handle_zern_test_start(self, mode = 0):
        """
        Handle start of closed-loop AO control test via Zernikes

        Args:
            mode = 0 - AO via Zernikes
            mode = 1 - Ao via slopes
        """
        self.control_zern_test(self.devices['sensor'], self.devices['mirror'], self.data_info, mode)

    def handle_zern_test_done(self):
        """
        Handle end of closed-loop AO control test via Zernikes
        """
        self.threads['zern_thread'].quit()
        self.threads['zern_thread'].wait()
        self.main.ui.ZernikeTestBtn.setChecked(False)

    def handle_zern_AO_start(self, mode = 1):
        """
        Handle start of closed-loop control process via Zernikes

        Args:
            mode = 1 - normal closed-loop AO process
            mode = 2 - closed-loop AO process with removal of obscured subaperture
            mode = 3 - closed-loop AO process with partial correction
            mode = 4 - full closed-loop AO process with removal of obscured subaperture and partial correction
        """
        self.control_zern_AO(self.devices['sensor'], self.devices['mirror'], self.data_info, mode)

    def handle_zern_AO_done(self, mode = 1):
        """
        Handle end of closed-loop control process via Zernikes
        """
        self.threads['zern_AO_thread'].quit()
        self.threads['zern_AO_thread'].wait()
        if mode == 1:
            self.main.ui.ZernikeAOBtn_1.setChecked(False)
        elif mode == 2:
            self.main.ui.ZernikeAOBtn_2.setChecked(False)
        elif mode == 3:
            self.main.ui.ZernikeAOBtn_3.setChecked(False)
        elif mode == 4:
            self.main.ui.ZernikeFullBtn.setChecked(False)

    def handle_slope_AO_start(self, mode = 1):
        """
        Handle start of closed-loop control process via slopes

        Args:
            mode = 1 - normal closed-loop AO process
            mode = 2 - closed-loop AO process with removal of obscured subaperture
            mode = 3 - closed-loop AO process with partial correction
            mode = 4 - full closed-loop AO process with removal of obscured subaperture and partial correction
        """
        self.control_slope_AO(self.devices['sensor'], self.devices['mirror'], self.data_info, mode)

    def handle_slope_AO_done(self, mode = 1):
        """
        Handle end of closed-loop control process via slopes
        """
        self.threads['slopes_AO_thread'].quit()
        self.threads['slopes_AO_thread'].wait()
        if mode == 1:
            self.main.ui.slopeAOBtn_1.setChecked(False)
        elif mode == 2:
            self.main.ui.slopeAOBtn_2.setChecked(False)
        elif mode == 3:
            self.main.ui.slopeAOBtn_3.setChecked(False)
        elif mode == 4:
            self.main.ui.SlopeFullBtn.setChecked(False)

    def handle_focus_start(self, AO_type = 0):
        """
        Handle start of remote focusing according to type of AO correction
        Args:
            AO_type = 0 - 'None'
            AO_type = 1 - 'Zernike AO 3'
            AO_type = 2 - 'Zernike Full'
            AO_type = 3 - 'Slope AO 3'
            AO_type = 4 - 'Slope Full'
        """
        if AO_type == 0:
            self.control_zern_AO(self.devices['sensor'], self.devices['mirror'], self.data_info, 5)
        elif AO_type in {1,2}:
            self.control_zern_AO(self.devices['sensor'], self.devices['mirror'], self.data_info, AO_type + 2)
        elif AO_type in {3,4}:
            self.control_slope_AO(self.devices['sensor'], self.devices['mirror'], self.data_info, AO_type)

    def handle_focus_done(self, mode = 0):
        """
        Handle end of remote focusing
        """
        try:
            self.threads['zern_AO_thread'].quit()
            self.threads['zern_AO_thread'].wait()
        except KeyError:
            pass
        try:
            self.threads['slopes_AO_thread'].quit()
            self.threads['slopes_AO_thread'].wait()
        except KeyError:
            pass
        if mode == 0:
            self.main.ui.moveBtn.setChecked(False)
        elif mode == 1:
            self.main.ui.scanBtn.setChecked(False)

    def stop(self):
        """
        Stop all workers, threads, and devices
        """
        # Stop workers and threads
        for worker in self.workers:
            self.workers[worker].stop()

        for thread in self.threads:
            self.threads[thread].quit()
            self.threads[thread].wait()

        # Stop and reset mirror instance
        # try:
        #    self.devices['mirror'].Stop()
        #    self.devices['mirror'].Reset()
        # except Exception as e:
        #     logger.warning("Error on mirror stop: {}".format(e))

        # # Stop sensor instance
        # try:
        #     self.devices['sensor'].stop_acquisition()
        # except Exception as e:
        #     logger.warning("Error on sensor stop: {}".format(e))

    def quit(self):
        """ 
        Quit application. 
        
        Clean up application, including shutting down workers, threads, and hardware devices.
        """
        # Stop workers and threads
        for worker in self.workers:
            self.workers[worker].stop()

        for thread in self.threads:
            self.threads[thread].quit()
            self.threads[thread].wait()

        # Stop and reset mirror instance
        try:
           self.devices['mirror'].Stop()
           self.devices['mirror'].Reset()
        except Exception as e:
            logger.warning("Error on mirror quit: {}".format(e))

        # Stop and close sensor instance
        try:
            self.devices['sensor'].stop_acquisition()
            self.devices['sensor'].close_device()
        except Exception as e:
            logger.warning("Error on sensor quit: {}".format(e))

        # Close other windows
        self.main.close()


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