# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SHViewer.ui',
# licensing of 'SHViewer.ui' applies.
#
# Created: Sun Nov 17 16:38:10 2019
#      by: pyside2-uic  running on PySide2 5.13.2
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_SHViewer(object):
    def setupUi(self, SHViewer):
        SHViewer.setObjectName("SHViewer")
        SHViewer.resize(1000, 850)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(SHViewer.sizePolicy().hasHeightForWidth())
        SHViewer.setSizePolicy(sizePolicy)
        SHViewer.setMinimumSize(QtCore.QSize(1000, 850))
        self.gridLayout = QtWidgets.QGridLayout(SHViewer)
        self.gridLayout.setObjectName("gridLayout")
        self.topLayout = QtWidgets.QVBoxLayout()
        self.topLayout.setSpacing(0)
        self.topLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.topLayout.setObjectName("topLayout")
        self.SHViewerLayout = QtWidgets.QHBoxLayout()
        self.SHViewerLayout.setObjectName("SHViewerLayout")
        self.graphicsView = ImageView(SHViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)
        self.graphicsView.setMinimumSize(QtCore.QSize(0, 0))
        self.graphicsView.setObjectName("graphicsView")
        self.SHViewerLayout.addWidget(self.graphicsView)
        self.topLayout.addLayout(self.SHViewerLayout)
        self.gridLayout.addLayout(self.topLayout, 2, 0, 1, 1)

        self.retranslateUi(SHViewer)
        QtCore.QMetaObject.connectSlotsByName(SHViewer)

    def retranslateUi(self, SHViewer):
        SHViewer.setWindowTitle(QtWidgets.QApplication.translate("SHViewer", "Form", None, -1))

from doptical.gui.common import ImageView

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SHViewer = QtWidgets.QWidget()
    ui = Ui_SHViewer()
    ui.setupUi(SHViewer)
    SHViewer.show()
    sys.exit(app.exec_())

