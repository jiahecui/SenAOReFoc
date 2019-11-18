from PySide2.QtWidgets import QApplication, QWidget, QGraphicsScene, QGraphicsView, QRubberBand
from PySide2.QtGui import QPixmap
from PySide2.QtCore import QSize, QPoint, QRect, Signal

import qtawesome as qta
from qimage2ndarray import array2qimage, gray2qimage

import numpy as np

from sensorbasedAO.gui.ui.SHViewer import Ui_SHViewer
from sensorbasedAO.data import Image
from sensorbasedAO.config import config

class SHViewer(QWidget):
    """
    SH viewer class
    """

    def __init__(self, parent = None, dtype = 'uint8'):
        super().__init__(parent)

        # Set up and initialise Ui_SHViewer class
        self.ui = Ui_SHViewer()
        self.ui.setupUi(self)

        # Initialise image array and datatype
        self.sensor_width = config['camera']['sensor_width']
        self.sensor_height = config['camera']['sensor_height']
        self.array_raw = Image(np.zeros(shape = (self.sensor_width, self.sensor_height)))
        self.array = Image(np.zeros(shape = (self.sensor_width, self.sensor_height)))
        self.dtype = dtype

        # Get image display settings
        self.settings = self.get_settings()

        # Display image on image viewer
        self.ui.graphicsView.setImage(array2qimage(self.array), reset = False)

    def get_settings(self):
        settings = {}
        settings['normalise'] = config['image']['normalise']
        settings['rescale'] = config['image']['rescale']
        settings['norm_min'] = config['image']['norm_min'] 
        settings['norm_max'] = config['image']['norm_max']
        settings['data_min'] = config['camera']['data_min']
        settings['data_max'] = config['camera']['data_max']

        return settings

    #==========Methods==========#
    def set_image(self, array, flag = 0):
        # Set raw data array
        self.array_raw = array

        # Update image display settings
        if flag:
            self.update()
        else:
            self.array = self.array_raw.copy()

        # Display image on image viewer
        self.ui.graphicsView.setImage(array2qimage(self.array))

    def update(self):
        self.array = self.array_raw.copy()

        print("Values in image array before update:", self.array)

        # Get image display settings
        settings = self.get_settings()

        # Get minimum and maximum pixel clipping value
        scale_min = np.interp(settings['norm_min'], \
            (np.iinfo(self.dtype).min, np.iinfo(self.dtype).max), \
                (settings['data_min'], settings['data_max']))
        scale_max = np.interp(settings['norm_max'], \
            (np.iinfo(self.dtype).min, np.iinfo(self.dtype).max), \
                (settings['data_min'], settings['data_max']))
            
        print('Scale_min and scale_max equals {} and {}'.format(scale_min, scale_max))

        # Clip array to scale_min and scale_max
        self.array = np.clip(self.array, scale_min, scale_max)

        print("Values in image array after normalise:", self.array)

        # Rescale to image viewer dtype
        if settings['rescale']:
            self.array = np.interp(self.array, (self.array.min(), self.array.max()), \
                (np.iinfo(self.dtype).min, np.iinfo(self.dtype).max)).astype(self.dtype)
        elif settings['normalise']:
            self.array = np.interp(self.array, (self.array.min(), self.array.max()), \
                (settings['norm_min'], settings['norm_max'])).astype(self.dtype)
        else:
            self.array = np.interp(self.array, (scale_min, scale_max), \
                (np.iinfo(self.dtype).min, np.iinfo(self.dtype).max)).astype(self.dtype)

        print("Values in image array after rescale:", self.array)


if __name__ == '__main__':
    import sys
 
    app = QApplication(sys.argv)
    SH_viewer = SHViewer()
    SH_viewer.show()
    SH_viewer.set_image(np.random.randint(0,10,size = (100, 100)))
    sys.exit(app.exec_())