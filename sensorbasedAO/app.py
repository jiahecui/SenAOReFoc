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
            'calibration_img': {}, 'AO_img': {}}

        # Initialise output HDF5 file
        # self.output_data = h5py.File('data_info.h5', 'a')
        # keys = list(self.data_info.keys())
        # grp1 = self.output_data.create_group(keys[0])
        # grp2 = self.output_data.create_group(keys[1])
        # grp3 = self.output_data.create_group(keys[2])
        # grp4 = self.output_data.create_group(keys[3])
        # grp5 = self.output_data.create_group(keys[4])
        # grp6 = self.output_data.create_group(keys[5])
        # grp7 = self.output_data.create_group(keys[6])
        # self.output_data.close()

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
        pos_worker.info.connect(lambda obj: self.handle_SB_info(obj))
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
               del grp4[k]
            grp4.create_dataset(k, data = v)
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