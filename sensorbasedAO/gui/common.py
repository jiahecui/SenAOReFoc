from PySide2.QtWidgets import QGraphicsScene, QGraphicsView
from PySide2.QtGui import QPixmap
from PySide2.QtCore import QPointF, Qt, Signal

class ImageView(QGraphicsView):
    clicked = Signal(QPointF)
    mouse_moved = Signal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.pixmap = None
        self.zoom_factor = 1

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.show()

    def setImage(self, image, reset=True):
        pixmap = QPixmap.fromImage(image)

        if self.pixmap:
            self.pixmap.setPixmap(pixmap)
        else:
            self.pixmap = self.scene.addPixmap(pixmap)

        if reset:
            self.reset()

    def reset(self):
        rect = self.pixmap.boundingRect()
        self.scene.setSceneRect(rect)
        self.fitInView(rect, Qt.KeepAspectRatio)

        self.zoom_factor = 1

    def zoom(self, factor):
        self.zoom_factor = factor
        self.scale(self.zoom_factor,self.zoom_factor)

