"""
Module for viewing S-H spots 
"""
from PySide2.QtWidgets import QApplication, QWidget, QGraphicsScene, QGraphicsView, QRubberBand
from PySide2.QtGui import QPixmap
from PySide2.QtCore import QSize, QPoint, QRect, Signal
from qimage2ndarray import array2qimage, gray2qimage
import qtawesome as qta
import numpy as np
from sensorbasedAO.gui.ui.SHViewer import Ui_SHViewer
from sensorbasedAO.data import Image
from sensorbasedAO.config import config

class SHViewer(QWidget):
    """
    SH viewer class
    """
    dock = Signal(bool)

    def __init__(self, parent = None, dtype = 'uint8'):
        super().__init__(parent)

        # Set up and initialise Ui_SHViewer class
        self.ui = Ui_SHViewer()
        self.ui.setupUi(self)
        self.customise_ui()

        # Initialise image array and datatype
        self.array_raw = Image(np.zeros(shape = (1,1)))
        self.array = Image(np.zeros(shape = (1, 1)))
        self.dtype = dtype

        # Default imageviewer docked
        self.docked = True

        # Bind handler
        self.ui.SHViewerDock.clicked.connect(self.on_dock)

        # Get image display settings
        self.settings = self.get_settings()

        # Update settings on display image
        self.update()

        # Display image on image viewer
        self.ui.graphicsView.setImage(array2qimage(self.array), reset = False)

    def customise_ui(self):
        self.ui.SHViewerDock.setIcon(qta.icon('ei.resize-full'))
        self.ui.SHViewerDock.setIconSize(QSize(14, 14)) 

    def get_settings(self):
        settings = {}
        settings['normalise'] = config['image']['normalise']
        settings['rescale'] = config['image']['rescale']
        settings['norm_min'] = config['image']['norm_min'] 
        settings['norm_max'] = config['image']['norm_max']
        settings['data_min'] = config['camera']['data_min']
        settings['data_max'] = config['camera']['data_max']

        return settings

    def on_dock(self):
        self.docked = not self.docked
        self.dock.emit(self.docked)

    #==========Methods==========#
    def set_image(self, array):
        # Set raw data array
        self.array_raw = array

        # Update image display settings
        self.update()

        print("size of image array before converted to qimage:", np.size(self.array))
        print("values in image array:", self.array)

        # Display image on image viewer
        self.ui.graphicsView.setImage(array2qimage(self.array))

    def update(self):
        self.array = self.array_raw.copy()

        # Get image display settings
        settings = self.get_settings()

        # Get minimum and maximum pixel clipping value
        scale_min = np.interp(settings['norm_min'], \
            (np.iinfo(self.dtype).min, np.iinfo(self.dtype).max), \
                (settings['data_min'], settings['data_max']))
        scale_max = np.interp(settings['norm_max'], \
            (np.iinfo(self.dtype).min, np.iinfo(self.dtype).max), \
                (settings['data_min'], settings['data_max']))

        # Clip array to scale_min and scale_max
        self.array = np.clip(self.array, scale_min, scale_max)

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


if __name__ == '__main__':
    import sys
 
    app = QApplication(sys.argv)
    SH_viewer = SHViewer()
    SH_viewer.show()
    SH_viewer.set_image(np.random.randint(0,10,size=(100,100)))
    sys.exit(app.exec_())