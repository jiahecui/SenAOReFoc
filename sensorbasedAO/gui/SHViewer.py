"""
Module for viewing S-H spots 
"""
from PySide2.QtWidgets import QApplication, QWidget, QGraphicsScene, QGraphicsView, QRubberBand
from PySide2.QtGui import QPixmap
from PySide2.QtCore import QSize, QPoint, QRect, Signal
from qimage2ndarray import array2qimage, gray2qimage
import numpy as np
from gui.ui.SHViewer import Ui_SHViewer
from doptical.config import config
import qtawesome as qta

class SHViewer(QWidget):
    """
    SH viewer class
    """
    dock = Signal(bool)

    self.ui = Ui_SHViewer()
    self.ui.setupUi(self)
    self.customise_ui()

    self.array = 