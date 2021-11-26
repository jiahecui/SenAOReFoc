from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QThread

import logging
import sys
import os
import argparse
import h5py
import numpy as np

import log
from config import config
from sensor import SENSOR
from mirror import MIRROR
from scanner import SCANNER
from server import SERVER
from gui.main import Main
from SB_geometry import Setup_SB
from SB_position import Positioning
from centroiding import Centroiding
from calibration import Calibration
from conversion import Conversion
from calibration_zern import Calibration_Zern
from calibration_RF import Calibration_RF
from data_collection import Data_Collection
from AO_zernikes import AO_Zernikes
from AO_slopes import AO_Slopes
from SH_acquisition import Acquisition

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
            'calibration_img': {}, 'calibration_RF_img': {}, 'calibration_RF_info': {}, 'AO_img': {}, 'focusing_info': {}}
        self.AO_group = {'data_collect': {}, 'zern_AO_1': {}, 'zern_AO_2': {}, 'zern_AO_3': {}, 'zern_AO_full': {}, \
            'slope_AO_1': {}, 'slope_AO_2': {}, 'slope_AO_3': {}, 'slope_AO_full': {}}

        # Initialise output HDF5 file
        self.HDF5_init()

        # Initialise workers and threads
        self.workers = {}
        self.threads = {}

        # Initialise background server
        self.servers = {}

        # Add devices
        self.devices = {}
        self.add_devices()

        # Open main GUI window
        self.main = Main(self, debug = debug)
        self.main.show()

        # Initialise remote focusing voltages
        self.remote_focus_voltages = h5py.File('exec_files/RF_calib_volts_interp_full_01um_1501.mat','r').get('interp_volts')
        self.remote_focus_voltages = np.array(self.remote_focus_voltages).T

    def add_server(self):
        """
        Add background server system
        """
        try:
            self.main.ui.stopRFSpin.setValue(1)
            server = SERVER.get(self, self.main, portNum = 18812)
        except Exception as e:
            logger.warning('Server initialisation error: {}'.format(e))
            server = None

        self.servers['server'] = server

        try:
            self.servers['server'].start()
        except Exception as e:
            logger.warning('Server start error: {}'.format(e))

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
                sensor.open_device_by_SN(config['camera']['SN'])
                print('Sensor load success.')
            except Exception as e:
                logger.warning('Sensor load error.', e)
                sensor = None

        self.devices['sensor'] = sensor

        # Start camera acquisition
        if not self.debug:
            self.devices['sensor'].start_acquisition()
        
        # Add deformable mirror
        if self.debug:
            mirror = MIRROR.get('debug')
        else:
            try:
                mirror = MIRROR.get(config['DM']['SN'])
                print('Mirror load success.')
            except Exception as e:
                logger.warning('Mirror load error.', e)
                mirror = None

        self.devices['mirror'] = mirror

        # Add scanner
        if config['data_collect']['zern_gen'] in [4, 5]:

            if self.debug:
                scanner = SCANNER.get('debug')
            else:
                try:
                    scanner = mtidevice.MTIDevice()
                    table = scanner.GetAvailableDevices()

                    if table.NumDevices == 0:
                        print('There are no devices available.')
                        return None

                    scanner.ListAvailableDevices(table)
                    portnumber = config['scanner']['portnumber']
                    
                    if os.name == 'nt':
                        portName = 'COM' + portnumber

                    scanner.ConnectDevice(portName)

                    # Initialise controller parameters
                    params = scanner.GetDeviceParams()
                    params.VdifferenceMax = 159
                    params.HardwareFilterBw = 500
                    params.Vbias = 80
                    params.SampleRate = 20000
                    scanner.SetDeviceParams(params)

                    # Set controller data mode
                    scanner.ResetDevicePosition()
                    scanner.StartDataStream()

                    # Turn the MEMS controller on
                    scanner.SetDeviceParam(MTIParam.MEMSDriverEnable, True)

                    print('Scanner load success.')
                except Exception as e:
                    logger.warning('Scanner load error', e)
                    scanner = None

            self.devices['scanner'] = scanner

    def stop_server(self):
        """
        Stop background server system
        """
        try:
            self.servers['server'].close()
        except Exception as e:
            logger.warning("Error on server stop: {}".format(e))
            return False
    
    def stop_focus(self):
        """
        Stop remote focusing during xz scan by signal of client software
        """
        try:
            self.main.ui.stopRFSpin.setValue(0)
            print()
            self.threads['zern_AO_thread'].quit()
            self.threads['zern_AO_thread'].wait()
            self.devices['mirror'].Reset()
            self.stop_server()
        except Exception as e:
            logger.warning("Error on RF stop: {}".format(e))
            raise

    def setup_SB(self, mirror):
        """
        Setup search block geometry and get reference centroids
        """
        # Create SB worker and thread
        SB_thread = QThread()
        SB_thread.setObjectName('SB_thread')
        SB_worker = Setup_SB(mirror)
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
        pos_worker = Positioning(sensor, SB_info, debug = self.debug)
        pos_worker.moveToThread(pos_thread)

        # Connect to signals
        pos_thread.started.connect(pos_worker.run)
        pos_worker.layer.connect(lambda obj: self.handle_layer_disp(obj))
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

    def get_centroids(self, sensor, mirror, SB_info):
        """
        Get actual centroids of S-H spots for system aberration correction
        """
        # Create centroiding worker and thread
        cent_thread = QThread()
        cent_thread.setObjectName('cent_thread')
        cent_worker = Centroiding(sensor, mirror, SB_info, debug = self.debug)
        cent_worker.moveToThread(cent_thread)

        # Connect to signals
        cent_thread.started.connect(cent_worker.run)
        cent_worker.layer.connect(lambda obj: self.handle_layer_disp(obj))
        cent_worker.message.connect(lambda obj: self.handle_message_disp(obj))
        cent_worker.SB_info.connect(lambda obj: self.handle_SB_info(obj))
        cent_worker.write.connect(self.write_SB_info)
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
        calib_worker = Calibration(sensor, mirror, SB_info, debug = self.debug)
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

    def control_data_collect(self, sensor, mirror, data_info, mode, scanner):
        """
        Perform automated data collection
        """
        # Create data collection worker and thread
        data_collect_thread = QThread()
        data_collect_thread.setObjectName('data_collect_thread')
        data_collect_worker = Data_Collection(sensor, mirror, data_info, scanner, debug = self.debug)
        data_collect_worker.moveToThread(data_collect_thread)

        # Connect to signals
        if mode == 0:
            data_collect_thread.started.connect(data_collect_worker.run0)
        elif mode == 1:
            data_collect_thread.started.connect(data_collect_worker.run1)
        elif mode == 2:
            data_collect_thread.started.connect(data_collect_worker.run2)
        elif mode == 3:
            data_collect_thread.started.connect(data_collect_worker.run3)
        elif mode == 4:
            data_collect_thread.started.connect(data_collect_worker.run4)
        elif mode == 5:
            data_collect_thread.started.connect(data_collect_worker.run5)

        data_collect_worker.done.connect(self.handle_data_collect_done)
        data_collect_worker.image.connect(lambda obj: self.handle_image_disp(obj))   
        data_collect_worker.message.connect(lambda obj: self.handle_message_disp(obj))
        data_collect_worker.info.connect(lambda obj: self.handle_AO_info(obj))
        data_collect_worker.write.connect(self.write_AO_info)
        data_collect_worker.error.connect(lambda obj: self.handle_error(obj))

        # Store data collection worker and thread
        self.workers['data_collect_worker'] = data_collect_worker
        self.threads['data_collect_thread'] = data_collect_thread

        # Start data collection thread
        data_collect_thread.start()

    def control_zern_AO(self, sensor, mirror, data_info, mode):
        """
        Closed-loop AO control via Zernikes + remote focusing
        """
        # Create Zernike AO worker and thread
        zern_AO_thread = QThread()
        zern_AO_thread.setObjectName('zern_AO_thread')
        zern_AO_worker = AO_Zernikes(sensor, mirror, data_info, self.main, debug = self.debug)
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
        elif mode == 6:
            zern_AO_thread.started.connect(zern_AO_worker.run6)
        elif mode == 7:
            zern_AO_thread.started.connect(zern_AO_worker.run7)

        zern_AO_worker.done.connect(lambda mode: self.handle_zern_AO_done(mode))
        zern_AO_worker.done2.connect(lambda mode: self.handle_focus_done(mode))
        zern_AO_worker.done3.connect(self.handle_tracking_done)
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
        slopes_AO_worker = AO_Slopes(sensor, mirror, data_info, debug = self.debug)
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

    def acq_SH_image(self, sensor, mode):
        """
        Acquire SH images
        """
        # Create SH image acquisition worker and thread
        acq_thread = QThread()
        acq_thread.setObjectName('acq_thread')
        acq_worker = Acquisition(sensor)
        acq_worker.moveToThread(acq_thread)

        # Connect to signals
        if mode == 0:
            acq_thread.started.connect(acq_worker.run0)
        if mode == 1:
            acq_thread.started.connect(acq_worker.run1)
        if mode == 2:
            acq_thread.started.connect(acq_worker.run2)

        acq_worker.done.connect(self.handle_acq_done)
        acq_worker.message.connect(lambda obj: self.handle_message_disp(obj))
        acq_worker.error.connect(lambda obj: self.handle_error(obj))
        acq_worker.image.connect(lambda obj: self.handle_image_disp(obj))

        # Store Zernike AO worker and thread
        self.workers['acq_worker'] = acq_worker
        self.threads['acq_thread'] = acq_thread

        # Start Zernike AO thread
        acq_thread.start()

    def calibrate_RF(self, sensor, mirror, data_info, direct):
        """
        Calibrate remote focusing
        """
        # Create calib_RF worker and thread
        calib_RF_thread = QThread()
        calib_RF_thread.setObjectName('calib_RF_thread')
        calib_RF_worker = Calibration_RF(sensor, mirror, data_info)
        calib_RF_worker.moveToThread(calib_RF_thread)

        # Connect to signals
        if direct == 0:
            calib_RF_thread.started.connect(calib_RF_worker.run0)
        if direct == 1:
            calib_RF_thread.started.connect(calib_RF_worker.run1)
            
        calib_RF_worker.image.connect(lambda obj: self.handle_image_disp(obj))
        calib_RF_worker.message.connect(lambda obj: self.handle_message_disp(obj))
        calib_RF_worker.info.connect(lambda obj: self.handle_mirror_info(obj))
        calib_RF_worker.write.connect(self.write_mirror_info)
        calib_RF_worker.calib_info.connect(lambda obj: self.handle_calibration_RF_info(obj))
        calib_RF_worker.calib_write.connect(self.write_calibration_RF_info)
        calib_RF_worker.error.connect(lambda obj: self.handle_error(obj))
        calib_RF_worker.done.connect(self.handle_calib_RF_done)

        # Store calib_RF worker and thread
        self.workers['calib_RF_worker'] = calib_RF_worker
        self.threads['calib_RF_thread'] = calib_RF_thread

        # Start calib_RF thread
        calib_RF_thread.start()

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
        self.main.update_image(obj, flag = 1, SB_settings = self.data_info['SB_info'])

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

    def handle_calibration_RF_info(self, obj):
        """
        Handle remote focusing calibration information
        """
        self.data_info['calibration_RF_info'].update(obj)

    def write_calibration_RF_info(self):
        """
        Write calibration RF info into HDF5 file
        """
        self.output_data = h5py.File('data_info.h5', 'a')
        grp5 = self.output_data['calibration_RF_info']
        for k, v in self.data_info['calibration_RF_info'].items():
            if k in grp5:
               del grp5[k]
            grp5.create_dataset(k, data = v)
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
        grp6 = self.output_data['focusing_info']
        for k, v in self.data_info['focusing_info'].items():
            if k in grp6:
               del grp6[k]
            grp6.create_dataset(k, data = v)
        self.output_data.close()

    def handle_error(self, error):
        """
        Handle errors from threads
        """
        raise(RuntimeError(error))

    def handle_SB_start(self):
        """
        Handle start of search block geometry setup
        """
        self.setup_SB(self.devices['mirror'])
        
    def handle_SB_done(self):
        """
        Handle end of search block geometry setup
        """  
        self.workers['SB_worker'].stop()    
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
        self.workers['pos_worker'].stop() 
        self.threads['pos_thread'].quit()
        self.threads['pos_thread'].wait()
        self.main.ui.positionBtn.setChecked(False)

    def handle_cent_start(self):
        """
        Handle start of S-H spot centroid calculation for system aberration calibration
        """
        self.get_centroids(self.devices['sensor'], self.devices['mirror'], self.data_info['SB_info'])

    def handle_cent_done(self):
        """
        Handle end of S-H spot centroid calculation
        """
        self.workers['cent_worker'].stop() 
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
        self.workers['calib_worker'].stop() 
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
        self.workers['conv_worker'].stop() 
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
        self.workers['calib2_worker'].stop() 
        self.threads['calib2_thread'].quit()
        self.threads['calib2_thread'].wait()
        self.main.ui.calibrateBtn_2.setChecked(False)

    def handle_data_collect_start(self, mode = 0):
        """
        Handle start of automated data collection

        Args:
            mode = 0 - Run closed-loop AO correction for each generated zernike mode aberration via Zernike control for 
                       all 'control_coeff_num' modes and multiple amplitudes of incremental steps
            mode = 1 - Run closed-loop AO correction for each generated zernike mode aberration via slopes control for 
                       all 'control_coeff_num' modes and multiple amplitudes of incremental steps
            mode = 2 - Run closed-loop AO correction for certain combinations of zernike mode aberrations via Zernike control
            mode = 3 - Run closed-loop AO correction for certain combinations of zernike mode aberrations via slopes control
        """
        self.control_data_collect(self.devices['sensor'], self.devices['mirror'], self.data_info, mode)

    def handle_data_collect_done(self):
        """
        Handle end of automated data collection
        """
        self.workers['data_collect_worker'].stop() 
        self.threads['data_collect_thread'].quit()
        self.threads['data_collect_thread'].wait()
        self.main.ui.DataCollectBtn.setChecked(False)

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
        self.workers['zern_AO_worker'].stop()
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
        self.workers['slopes_AO_worker'].stop()
        self.threads['slopes_AO_thread'].quit()
        self.threads['slopes_AO_thread'].wait()
        if mode == 1:
            self.main.ui.slopeAOBtn_1.setChecked(False)
        elif mode == 2:
            self.main.ui.slopeAOBtn_2.setChecked(False)
        elif mode == 3:
            self.main.ui.slopeAOBtn_3.setChecked(False)
        elif mode == 4:
            self.main.ui.slopeFullBtn.setChecked(False)

    def handle_acq_live_start(self):
        """
        Handle start of live acquisition of SHWS
        """
        self.acq_SH_image(self.devices['sensor'], mode = 0)

    def handle_acq_burst_start(self):
        """
        Handle start of burst acquisition of SHWS
        """
        self.acq_SH_image(self.devices['sensor'], mode = 1)

    def handle_acq_single_start(self):
        """
        Handle start of single acquisition of SHWS
        """
        self.acq_SH_image(self.devices['sensor'], mode = 2)

    def stop_acq(self):
        """
        Stops SHWS image acquisition
        """
        try:
            self.workers['acq_worker'].stop()
            self.threads['acq_thread'].quit()
            self.threads['acq_thread'].wait()
        except KeyError:
            pass
        except:
            raise

    def handle_acq_done(self):
        """
        Handle end of SHWS image acquisition
        """
        self.workers['acq_worker'].stop()
        self.threads['acq_thread'].quit()
        self.threads['acq_thread'].wait()
        self.main.ui.liveAcqBtn.setChecked(False)
        self.main.ui.burstAcqBtn.setChecked(False)
        self.main.ui.singleAcqBtn.setChecked(False)

    def handle_DM_reset(self):
        """
        Handle reset of deformable mirror
        """
        try:
            self.devices['mirror'].Reset()
            self.main.ui.DMRstBtn.setChecked(False)
            print('DM reset success.')
        except Exception as e:
            logger.warning("Error on mirror reset: {}".format(e))

    def handle_scanner_reset(self):
        """
        Handle reset of scanner
        """
        try:
            self.devices['scanner'].ResetDevicePosition()
            self.main.ui.scannerRstBtn.setChecked(False)
            print('Scanner reset success.')
        except Exception as e:
            logger.warning("Error on scanner reset: {}".format(e))

    def handle_camera_expo(self, camera_expo):
        """
        Handle update of camera exposure
        """
        try:
            self.devices['sensor'].set_exposure(camera_expo)
            print('Camera exposure set success.')
        except Exception as e:
            logger.warning("Error on setting camera exposure: {}".format(e))

    def handle_calib_RF_start(self, direct = 1):
        """
        Handle calibration of remote focusing
        """
        self.calibrate_RF(self.devices['sensor'], self.devices['mirror'], self.data_info, direct)

    def handle_calib_RF_done(self):
        """
        Handle calibration of remote focusing
        """
        self.workers['calib_RF_worker'].stop()
        self.threads['calib_RF_thread'].quit()
        self.threads['calib_RF_thread'].wait()
        self.main.ui.calibrateRFBtn.setChecked(False)

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
        if AO_type == 0 and self.data_info['focusing_info']['is_xz_scan'] == 0:
            self.control_zern_AO(self.devices['sensor'], self.devices['mirror'], self.data_info, 5)
        elif AO_type == 0 and self.data_info['focusing_info']['is_xz_scan'] == 1:
            self.control_zern_AO(self.devices['sensor'], self.devices['mirror'], self.data_info, 7)
        elif AO_type in {1,2}:
            self.control_zern_AO(self.devices['sensor'], self.devices['mirror'], self.data_info, AO_type + 2)
        elif AO_type in {3,4}:
            self.control_slope_AO(self.devices['sensor'], self.devices['mirror'], self.data_info, AO_type)

    def handle_focus_done(self, mode = 0):
        """
        Handle end of remote focusing
        """
        try:
            self.workers['zern_AO_worker'].stop()
            self.threads['zern_AO_thread'].quit()
            self.threads['zern_AO_thread'].wait()
        except KeyError:
            pass
        try:
            self.workers['slopes_AO_worker'].stop()
            self.threads['slopes_AO_thread'].quit()
            self.threads['slopes_AO_thread'].wait()
        except KeyError:
            pass
        if mode == 0:
            self.main.ui.moveBtn.setChecked(False)
        elif mode == 1:
            self.main.ui.scanBtn.setChecked(False)
        else:
            pass

    def handle_tracking_start(self):
        """
        Handle start of surface tracking
        """
        self.control_zern_AO(self.devices['sensor'], self.devices['mirror'], self.data_info, 6)

    def handle_tracking_done(self):
        """
        Handle end of surface tracking
        """
        print()
        self.workers['zern_AO_worker'].stop()
        self.threads['zern_AO_thread'].quit()
        self.threads['zern_AO_thread'].wait()
        # self.devices['mirror'].Stop()
        # self.devices['mirror'].Reset()
        self.main.ui.trackBtn.setChecked(False)
        print('Surface tracking function stopped.')

    def handle_RF_control(self, val = 0):
        """
        Handle remote focusing control slider
        """
        # Update value of remote focusing position 
        RF_index = int(val // config['RF']['step_incre']) + config['RF']['index_offset']
        voltages_defoc = np.ravel(self.remote_focus_voltages[:, RF_index])

        # Apply remote focusing voltages
        voltages = config['DM']['vol_bias'] + voltages_defoc

        # Send voltages to mirror
        self.devices['mirror'].Send(voltages)

        print('Focus position: {} um'.format(val))       

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
        try:
           self.devices['mirror'].Stop()
           self.devices['mirror'].Reset()
        except Exception as e:
            logger.warning("Error on mirror stop: {}".format(e))

        # Stop sensor instance
        try:
            self.devices['sensor'].stop_acquisition()
        except Exception as e:
            logger.warning("Error on sensor stop: {}".format(e))

        # Stop scanner instance
        try:
            self.devices['scanner'].ResetDevicePosition()
            self.devices['scanner'].StopDataStream()
        except Exception as e:
            logger.warning("Error on scanner stop: {}".format(e))    

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
        if not self.debug:
            try:
                self.devices['mirror'].Stop()
                self.devices['mirror'].Reset()
            except Exception as e:
                logger.warning("Error on mirror quit: {}".format(e))

        # Stop and close sensor instance
        if not self.debug:
            try:
                self.devices['sensor'].stop_acquisition()
                self.devices['sensor'].close_device()
            except Exception as e:
                logger.warning("Error on sensor quit: {}".format(e))

        # Stop and reset scanner instance
        if not self.debug:
            try:
                self.devices['scanner'].ResetDevicePosition()
                self.devices['scanner'].StopDataStream()
                self.devices['scanner'].SetDeviceParam(MTIParam.MEMSDriverEnable, False)
                self.devices['scanner'].DisconnectDevice()
            except Exception as e:
                logger.warning("Error on scanner quit: {}".format(e))

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