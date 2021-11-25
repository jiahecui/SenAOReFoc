from PySide2.QtWidgets import QApplication, QWidget, QGraphicsScene, QGraphicsView, QVBoxLayout
from PySide2.QtGui import QPixmap, QTransform
from PySide2.QtCore import QRectF, QPointF, Qt, Signal

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

    def mousePressEvent(self, event):
        self.clicked.emit(self.mapToScene(event.pos()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

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

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            zoom = 1.1
        else:
            zoom = 0.9

        self.zoom(zoom)

    def zoom(self, factor):
        self.zoom_factor = factor
        self.scale(self.zoom_factor,self.zoom_factor)

class FloatingWidget(QWidget):
    def __init__(self, child_widget):
        super().__init__(None)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(child_widget)
        self.setLayout(self.vbox)
        self.setWindowFlags(Qt.CustomizeWindowHint |
                            Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.show()

