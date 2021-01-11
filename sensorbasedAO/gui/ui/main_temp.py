# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from sensorbasedAO.gui.SHViewer import SHViewer


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1580, 1180)
        sizePolicy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QSize(1580, 1180))
        icon = QIcon()
        icon.addFile(u"../resources/icons/AO.ico", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.mainLayout = QHBoxLayout()
        self.mainLayout.setObjectName(u"mainLayout")
        self.mainContentLayout = QVBoxLayout()
        self.mainContentLayout.setObjectName(u"mainContentLayout")
        self.horizontalLayout_1 = QHBoxLayout()
        self.horizontalLayout_1.setObjectName(u"horizontalLayout_1")
        self.initialiseBtn = QPushButton(self.centralwidget)
        self.initialiseBtn.setObjectName(u"initialiseBtn")
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(100)
        sizePolicy1.setVerticalStretch(30)
        sizePolicy1.setHeightForWidth(self.initialiseBtn.sizePolicy().hasHeightForWidth())
        self.initialiseBtn.setSizePolicy(sizePolicy1)
        self.initialiseBtn.setMinimumSize(QSize(150, 30))
        font = QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.initialiseBtn.setFont(font)
        self.initialiseBtn.setCheckable(True)

        self.horizontalLayout_1.addWidget(self.initialiseBtn)

        self.positionBtn = QPushButton(self.centralwidget)
        self.positionBtn.setObjectName(u"positionBtn")
        sizePolicy1.setHeightForWidth(self.positionBtn.sizePolicy().hasHeightForWidth())
        self.positionBtn.setSizePolicy(sizePolicy1)
        self.positionBtn.setMinimumSize(QSize(150, 30))
        self.positionBtn.setFont(font)
        self.positionBtn.setCheckable(True)

        self.horizontalLayout_1.addWidget(self.positionBtn)

        self.centroidBtn = QPushButton(self.centralwidget)
        self.centroidBtn.setObjectName(u"centroidBtn")
        sizePolicy1.setHeightForWidth(self.centroidBtn.sizePolicy().hasHeightForWidth())
        self.centroidBtn.setSizePolicy(sizePolicy1)
        self.centroidBtn.setMinimumSize(QSize(150, 30))
        self.centroidBtn.setFont(font)
        self.centroidBtn.setCheckable(True)

        self.horizontalLayout_1.addWidget(self.centroidBtn)


        self.mainContentLayout.addLayout(self.horizontalLayout_1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.calibrateBtn = QPushButton(self.centralwidget)
        self.calibrateBtn.setObjectName(u"calibrateBtn")
        sizePolicy1.setHeightForWidth(self.calibrateBtn.sizePolicy().hasHeightForWidth())
        self.calibrateBtn.setSizePolicy(sizePolicy1)
        self.calibrateBtn.setMinimumSize(QSize(150, 30))
        self.calibrateBtn.setFont(font)
        self.calibrateBtn.setCheckable(True)

        self.horizontalLayout_2.addWidget(self.calibrateBtn)

        self.conversionBtn = QPushButton(self.centralwidget)
        self.conversionBtn.setObjectName(u"conversionBtn")
        sizePolicy1.setHeightForWidth(self.conversionBtn.sizePolicy().hasHeightForWidth())
        self.conversionBtn.setSizePolicy(sizePolicy1)
        self.conversionBtn.setMinimumSize(QSize(150, 30))
        self.conversionBtn.setFont(font)
        self.conversionBtn.setCheckable(True)

        self.horizontalLayout_2.addWidget(self.conversionBtn)

        self.calibrateBtn_2 = QPushButton(self.centralwidget)
        self.calibrateBtn_2.setObjectName(u"calibrateBtn_2")
        sizePolicy1.setHeightForWidth(self.calibrateBtn_2.sizePolicy().hasHeightForWidth())
        self.calibrateBtn_2.setSizePolicy(sizePolicy1)
        self.calibrateBtn_2.setMinimumSize(QSize(150, 30))
        self.calibrateBtn_2.setFont(font)
        self.calibrateBtn_2.setCheckable(True)

        self.horizontalLayout_2.addWidget(self.calibrateBtn_2)


        self.mainContentLayout.addLayout(self.horizontalLayout_2)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.ZernikeArrEdt = QLineEdit(self.centralwidget)
        self.ZernikeArrEdt.setObjectName(u"ZernikeArrEdt")
        self.ZernikeArrEdt.setMinimumSize(QSize(0, 20))
        font1 = QFont()
        font1.setPointSize(10)
        self.ZernikeArrEdt.setFont(font1)

        self.gridLayout.addWidget(self.ZernikeArrEdt, 2, 0, 1, 1)

        self.ZernikeCoeffSpin = QSpinBox(self.centralwidget)
        self.ZernikeCoeffSpin.setObjectName(u"ZernikeCoeffSpin")
        self.ZernikeCoeffSpin.setMinimumSize(QSize(0, 20))
        self.ZernikeCoeffSpin.setFont(font1)
        self.ZernikeCoeffSpin.setMinimum(1)
        self.ZernikeCoeffSpin.setMaximum(200)
        self.ZernikeCoeffSpin.setValue(1)
        self.ZernikeCoeffSpin.setDisplayIntegerBase(10)

        self.gridLayout.addWidget(self.ZernikeCoeffSpin, 1, 0, 1, 1)

        self.ZernikeLbl = QLabel(self.centralwidget)
        self.ZernikeLbl.setObjectName(u"ZernikeLbl")
        self.ZernikeLbl.setFont(font)

        self.gridLayout.addWidget(self.ZernikeLbl, 0, 0, 1, 1)

        self.ZernikeOKBtn = QPushButton(self.centralwidget)
        self.ZernikeOKBtn.setObjectName(u"ZernikeOKBtn")
        self.ZernikeOKBtn.setMinimumSize(QSize(0, 20))
        self.ZernikeOKBtn.setCheckable(True)

        self.gridLayout.addWidget(self.ZernikeOKBtn, 2, 1, 1, 1)

        self.ZernikeValSpin = QDoubleSpinBox(self.centralwidget)
        self.ZernikeValSpin.setObjectName(u"ZernikeValSpin")
        self.ZernikeValSpin.setMinimumSize(QSize(0, 20))
        self.ZernikeValSpin.setFont(font1)
        self.ZernikeValSpin.setDecimals(3)
        self.ZernikeValSpin.setMinimum(-99.989999999999995)
        self.ZernikeValSpin.setSingleStep(0.010000000000000)
        self.ZernikeValSpin.setValue(0.000000000000000)

        self.gridLayout.addWidget(self.ZernikeValSpin, 1, 1, 1, 1)


        self.mainContentLayout.addLayout(self.gridLayout)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.ZernikeAOBtn_3 = QPushButton(self.centralwidget)
        self.ZernikeAOBtn_3.setObjectName(u"ZernikeAOBtn_3")
        sizePolicy1.setHeightForWidth(self.ZernikeAOBtn_3.sizePolicy().hasHeightForWidth())
        self.ZernikeAOBtn_3.setSizePolicy(sizePolicy1)
        self.ZernikeAOBtn_3.setMinimumSize(QSize(100, 30))
        self.ZernikeAOBtn_3.setFont(font1)
        self.ZernikeAOBtn_3.setCheckable(True)

        self.gridLayout_2.addWidget(self.ZernikeAOBtn_3, 2, 0, 1, 1)

        self.slopeAOBtn_3 = QPushButton(self.centralwidget)
        self.slopeAOBtn_3.setObjectName(u"slopeAOBtn_3")
        sizePolicy1.setHeightForWidth(self.slopeAOBtn_3.sizePolicy().hasHeightForWidth())
        self.slopeAOBtn_3.setSizePolicy(sizePolicy1)
        self.slopeAOBtn_3.setMinimumSize(QSize(100, 30))
        self.slopeAOBtn_3.setFont(font1)
        self.slopeAOBtn_3.setCheckable(True)

        self.gridLayout_2.addWidget(self.slopeAOBtn_3, 2, 1, 1, 1)

        self.ZernikeAOBtn_2 = QPushButton(self.centralwidget)
        self.ZernikeAOBtn_2.setObjectName(u"ZernikeAOBtn_2")
        sizePolicy1.setHeightForWidth(self.ZernikeAOBtn_2.sizePolicy().hasHeightForWidth())
        self.ZernikeAOBtn_2.setSizePolicy(sizePolicy1)
        self.ZernikeAOBtn_2.setMinimumSize(QSize(100, 30))
        self.ZernikeAOBtn_2.setFont(font1)
        self.ZernikeAOBtn_2.setCheckable(True)

        self.gridLayout_2.addWidget(self.ZernikeAOBtn_2, 1, 0, 1, 1)

        self.slopeAOBtn_2 = QPushButton(self.centralwidget)
        self.slopeAOBtn_2.setObjectName(u"slopeAOBtn_2")
        sizePolicy1.setHeightForWidth(self.slopeAOBtn_2.sizePolicy().hasHeightForWidth())
        self.slopeAOBtn_2.setSizePolicy(sizePolicy1)
        self.slopeAOBtn_2.setMinimumSize(QSize(100, 30))
        self.slopeAOBtn_2.setFont(font1)
        self.slopeAOBtn_2.setCheckable(True)

        self.gridLayout_2.addWidget(self.slopeAOBtn_2, 1, 1, 1, 1)

        self.ZernikeAOBtn_1 = QPushButton(self.centralwidget)
        self.ZernikeAOBtn_1.setObjectName(u"ZernikeAOBtn_1")
        sizePolicy1.setHeightForWidth(self.ZernikeAOBtn_1.sizePolicy().hasHeightForWidth())
        self.ZernikeAOBtn_1.setSizePolicy(sizePolicy1)
        self.ZernikeAOBtn_1.setMinimumSize(QSize(100, 30))
        self.ZernikeAOBtn_1.setFont(font1)
        self.ZernikeAOBtn_1.setCheckable(True)

        self.gridLayout_2.addWidget(self.ZernikeAOBtn_1, 0, 0, 1, 1)

        self.slopeAOBtn_1 = QPushButton(self.centralwidget)
        self.slopeAOBtn_1.setObjectName(u"slopeAOBtn_1")
        sizePolicy1.setHeightForWidth(self.slopeAOBtn_1.sizePolicy().hasHeightForWidth())
        self.slopeAOBtn_1.setSizePolicy(sizePolicy1)
        self.slopeAOBtn_1.setMinimumSize(QSize(100, 30))
        self.slopeAOBtn_1.setFont(font1)
        self.slopeAOBtn_1.setCheckable(True)

        self.gridLayout_2.addWidget(self.slopeAOBtn_1, 0, 1, 1, 1)

        self.ZernikeTestBtn = QPushButton(self.centralwidget)
        self.ZernikeTestBtn.setObjectName(u"ZernikeTestBtn")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(130)
        sizePolicy2.setVerticalStretch(30)
        sizePolicy2.setHeightForWidth(self.ZernikeTestBtn.sizePolicy().hasHeightForWidth())
        self.ZernikeTestBtn.setSizePolicy(sizePolicy2)
        self.ZernikeTestBtn.setMinimumSize(QSize(130, 30))
        self.ZernikeTestBtn.setFont(font1)
        self.ZernikeTestBtn.setCheckable(True)

        self.gridLayout_2.addWidget(self.ZernikeTestBtn, 2, 2, 1, 1)

        self.ZernikeFullBtn = QPushButton(self.centralwidget)
        self.ZernikeFullBtn.setObjectName(u"ZernikeFullBtn")
        sizePolicy2.setHeightForWidth(self.ZernikeFullBtn.sizePolicy().hasHeightForWidth())
        self.ZernikeFullBtn.setSizePolicy(sizePolicy2)
        self.ZernikeFullBtn.setMinimumSize(QSize(130, 30))
        self.ZernikeFullBtn.setFont(font1)
        self.ZernikeFullBtn.setCheckable(True)

        self.gridLayout_2.addWidget(self.ZernikeFullBtn, 0, 2, 1, 1)

        self.slopeFullBtn = QPushButton(self.centralwidget)
        self.slopeFullBtn.setObjectName(u"slopeFullBtn")
        sizePolicy2.setHeightForWidth(self.slopeFullBtn.sizePolicy().hasHeightForWidth())
        self.slopeFullBtn.setSizePolicy(sizePolicy2)
        self.slopeFullBtn.setMinimumSize(QSize(130, 30))
        self.slopeFullBtn.setFont(font1)
        self.slopeFullBtn.setCheckable(True)

        self.gridLayout_2.addWidget(self.slopeFullBtn, 1, 2, 1, 1)


        self.mainContentLayout.addLayout(self.gridLayout_2)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.imageAcqLbl = QLabel(self.centralwidget)
        self.imageAcqLbl.setObjectName(u"imageAcqLbl")
        sizePolicy3 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.imageAcqLbl.sizePolicy().hasHeightForWidth())
        self.imageAcqLbl.setSizePolicy(sizePolicy3)
        self.imageAcqLbl.setMinimumSize(QSize(170, 30))
        self.imageAcqLbl.setFont(font1)

        self.horizontalLayout_4.addWidget(self.imageAcqLbl)

        self.liveAcqBtn = QPushButton(self.centralwidget)
        self.liveAcqBtn.setObjectName(u"liveAcqBtn")
        sizePolicy1.setHeightForWidth(self.liveAcqBtn.sizePolicy().hasHeightForWidth())
        self.liveAcqBtn.setSizePolicy(sizePolicy1)
        self.liveAcqBtn.setMinimumSize(QSize(90, 30))
        self.liveAcqBtn.setFont(font1)
        self.liveAcqBtn.setCheckable(True)

        self.horizontalLayout_4.addWidget(self.liveAcqBtn)

        self.burstAcqBtn = QPushButton(self.centralwidget)
        self.burstAcqBtn.setObjectName(u"burstAcqBtn")
        sizePolicy1.setHeightForWidth(self.burstAcqBtn.sizePolicy().hasHeightForWidth())
        self.burstAcqBtn.setSizePolicy(sizePolicy1)
        self.burstAcqBtn.setMinimumSize(QSize(100, 30))
        self.burstAcqBtn.setFont(font1)
        self.burstAcqBtn.setCheckable(True)

        self.horizontalLayout_4.addWidget(self.burstAcqBtn)

        self.singleAcqBtn = QPushButton(self.centralwidget)
        self.singleAcqBtn.setObjectName(u"singleAcqBtn")
        sizePolicy1.setHeightForWidth(self.singleAcqBtn.sizePolicy().hasHeightForWidth())
        self.singleAcqBtn.setSizePolicy(sizePolicy1)
        self.singleAcqBtn.setMinimumSize(QSize(100, 30))
        self.singleAcqBtn.setFont(font1)
        self.singleAcqBtn.setCheckable(True)

        self.horizontalLayout_4.addWidget(self.singleAcqBtn)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer)


        self.mainContentLayout.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.miscellaneousLbl = QLabel(self.centralwidget)
        self.miscellaneousLbl.setObjectName(u"miscellaneousLbl")
        sizePolicy3.setHeightForWidth(self.miscellaneousLbl.sizePolicy().hasHeightForWidth())
        self.miscellaneousLbl.setSizePolicy(sizePolicy3)
        self.miscellaneousLbl.setMinimumSize(QSize(170, 30))
        self.miscellaneousLbl.setFont(font1)

        self.horizontalLayout_5.addWidget(self.miscellaneousLbl)

        self.DMRstBtn = QPushButton(self.centralwidget)
        self.DMRstBtn.setObjectName(u"DMRstBtn")
        sizePolicy1.setHeightForWidth(self.DMRstBtn.sizePolicy().hasHeightForWidth())
        self.DMRstBtn.setSizePolicy(sizePolicy1)
        self.DMRstBtn.setMinimumSize(QSize(90, 30))
        self.DMRstBtn.setFont(font1)
        self.DMRstBtn.setCheckable(True)

        self.horizontalLayout_5.addWidget(self.DMRstBtn)

        self.scannerRstBtn = QPushButton(self.centralwidget)
        self.scannerRstBtn.setObjectName(u"scannerRstBtn")
        sizePolicy1.setHeightForWidth(self.scannerRstBtn.sizePolicy().hasHeightForWidth())
        self.scannerRstBtn.setSizePolicy(sizePolicy1)
        self.scannerRstBtn.setMinimumSize(QSize(100, 30))
        self.scannerRstBtn.setFont(font1)
        self.scannerRstBtn.setCheckable(True)

        self.horizontalLayout_5.addWidget(self.scannerRstBtn)

        self.cameraExpoSpin = QSpinBox(self.centralwidget)
        self.cameraExpoSpin.setObjectName(u"cameraExpoSpin")
        sizePolicy3.setHeightForWidth(self.cameraExpoSpin.sizePolicy().hasHeightForWidth())
        self.cameraExpoSpin.setSizePolicy(sizePolicy3)
        self.cameraExpoSpin.setMinimumSize(QSize(50, 30))
        self.cameraExpoSpin.setFont(font1)
        self.cameraExpoSpin.setMinimum(30)
        self.cameraExpoSpin.setMaximum(1000000)
        self.cameraExpoSpin.setSingleStep(10)
        self.cameraExpoSpin.setValue(40000)

        self.horizontalLayout_5.addWidget(self.cameraExpoSpin)

        self.loopMaxSpin = QSpinBox(self.centralwidget)
        self.loopMaxSpin.setObjectName(u"loopMaxSpin")
        sizePolicy3.setHeightForWidth(self.loopMaxSpin.sizePolicy().hasHeightForWidth())
        self.loopMaxSpin.setSizePolicy(sizePolicy3)
        self.loopMaxSpin.setMinimumSize(QSize(50, 30))
        self.loopMaxSpin.setFont(font1)
        self.loopMaxSpin.setMinimum(1)
        self.loopMaxSpin.setMaximum(100)

        self.horizontalLayout_5.addWidget(self.loopMaxSpin)


        self.mainContentLayout.addLayout(self.horizontalLayout_5)

        self.remoteFocusLbl = QLabel(self.centralwidget)
        self.remoteFocusLbl.setObjectName(u"remoteFocusLbl")
        sizePolicy4 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.remoteFocusLbl.sizePolicy().hasHeightForWidth())
        self.remoteFocusLbl.setSizePolicy(sizePolicy4)
        font2 = QFont()
        font2.setPointSize(11)
        self.remoteFocusLbl.setFont(font2)
        self.remoteFocusLbl.setFrameShape(QFrame.WinPanel)
        self.remoteFocusLbl.setFrameShadow(QFrame.Sunken)
        self.remoteFocusLbl.setLineWidth(2)
        self.remoteFocusLbl.setAlignment(Qt.AlignCenter)

        self.mainContentLayout.addWidget(self.remoteFocusLbl)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.scanFocusCheck = QCheckBox(self.centralwidget)
        self.scanFocusCheck.setObjectName(u"scanFocusCheck")
        sizePolicy5 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.scanFocusCheck.sizePolicy().hasHeightForWidth())
        self.scanFocusCheck.setSizePolicy(sizePolicy5)
        font3 = QFont()
        font3.setPointSize(10)
        font3.setStrikeOut(False)
        self.scanFocusCheck.setFont(font3)
        self.scanFocusCheck.setLayoutDirection(Qt.LeftToRight)

        self.verticalLayout_3.addWidget(self.scanFocusCheck)

        self.focusDepthLbl = QLabel(self.centralwidget)
        self.focusDepthLbl.setObjectName(u"focusDepthLbl")
        self.focusDepthLbl.setFont(font1)

        self.verticalLayout_3.addWidget(self.focusDepthLbl)

        self.focusDepthSpin = QDoubleSpinBox(self.centralwidget)
        self.focusDepthSpin.setObjectName(u"focusDepthSpin")
        self.focusDepthSpin.setMinimumSize(QSize(0, 20))
        self.focusDepthSpin.setFont(font1)
        self.focusDepthSpin.setDecimals(1)
        self.focusDepthSpin.setMinimum(-100.000000000000000)
        self.focusDepthSpin.setMaximum(100.000000000000000)
        self.focusDepthSpin.setSingleStep(0.100000000000000)

        self.verticalLayout_3.addWidget(self.focusDepthSpin)

        self.stepIncreLbl = QLabel(self.centralwidget)
        self.stepIncreLbl.setObjectName(u"stepIncreLbl")
        self.stepIncreLbl.setFont(font1)

        self.verticalLayout_3.addWidget(self.stepIncreLbl)

        self.stepIncreSpin = QDoubleSpinBox(self.centralwidget)
        self.stepIncreSpin.setObjectName(u"stepIncreSpin")
        self.stepIncreSpin.setMinimumSize(QSize(0, 20))
        self.stepIncreSpin.setFont(font1)
        self.stepIncreSpin.setDecimals(1)
        self.stepIncreSpin.setMinimum(-100.000000000000000)
        self.stepIncreSpin.setMaximum(100.000000000000000)
        self.stepIncreSpin.setSingleStep(0.100000000000000)

        self.verticalLayout_3.addWidget(self.stepIncreSpin)

        self.stepNumLbl = QLabel(self.centralwidget)
        self.stepNumLbl.setObjectName(u"stepNumLbl")
        self.stepNumLbl.setFont(font1)

        self.verticalLayout_3.addWidget(self.stepNumLbl)

        self.stepNumSpin = QDoubleSpinBox(self.centralwidget)
        self.stepNumSpin.setObjectName(u"stepNumSpin")
        self.stepNumSpin.setMinimumSize(QSize(0, 20))
        self.stepNumSpin.setFont(font1)
        self.stepNumSpin.setDecimals(0)
        self.stepNumSpin.setMaximum(2000.000000000000000)

        self.verticalLayout_3.addWidget(self.stepNumSpin)

        self.startDepthLbl = QLabel(self.centralwidget)
        self.startDepthLbl.setObjectName(u"startDepthLbl")
        self.startDepthLbl.setFont(font1)

        self.verticalLayout_3.addWidget(self.startDepthLbl)

        self.startDepthSpin = QDoubleSpinBox(self.centralwidget)
        self.startDepthSpin.setObjectName(u"startDepthSpin")
        self.startDepthSpin.setMinimumSize(QSize(0, 20))
        self.startDepthSpin.setFont(font1)
        self.startDepthSpin.setDecimals(1)
        self.startDepthSpin.setMinimum(-100.000000000000000)
        self.startDepthSpin.setMaximum(100.000000000000000)
        self.startDepthSpin.setSingleStep(0.100000000000000)

        self.verticalLayout_3.addWidget(self.startDepthSpin)

        self.pauseTimeLbl = QLabel(self.centralwidget)
        self.pauseTimeLbl.setObjectName(u"pauseTimeLbl")
        self.pauseTimeLbl.setFont(font1)

        self.verticalLayout_3.addWidget(self.pauseTimeLbl)

        self.pauseTimeSpin = QDoubleSpinBox(self.centralwidget)
        self.pauseTimeSpin.setObjectName(u"pauseTimeSpin")
        self.pauseTimeSpin.setMinimumSize(QSize(0, 20))
        self.pauseTimeSpin.setFont(font1)
        self.pauseTimeSpin.setDecimals(2)
        self.pauseTimeSpin.setMaximum(1000.000000000000000)
        self.pauseTimeSpin.setSingleStep(0.010000000000000)

        self.verticalLayout_3.addWidget(self.pauseTimeSpin)


        self.horizontalLayout_3.addLayout(self.verticalLayout_3)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.AOTypeLbl = QLabel(self.centralwidget)
        self.AOTypeLbl.setObjectName(u"AOTypeLbl")
        sizePolicy6 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.AOTypeLbl.sizePolicy().hasHeightForWidth())
        self.AOTypeLbl.setSizePolicy(sizePolicy6)
        self.AOTypeLbl.setFont(font1)

        self.verticalLayout_5.addWidget(self.AOTypeLbl)

        self.AOTypeCombo = QComboBox(self.centralwidget)
        self.AOTypeCombo.addItem("")
        self.AOTypeCombo.addItem("")
        self.AOTypeCombo.addItem("")
        self.AOTypeCombo.addItem("")
        self.AOTypeCombo.addItem("")
        self.AOTypeCombo.setObjectName(u"AOTypeCombo")
        self.AOTypeCombo.setMinimumSize(QSize(0, 20))
        self.AOTypeCombo.setFont(font1)
        self.AOTypeCombo.setMouseTracking(True)
        self.AOTypeCombo.setEditable(False)

        self.verticalLayout_5.addWidget(self.AOTypeCombo)

        self.calibrateRFBtn = QPushButton(self.centralwidget)
        self.calibrateRFBtn.setObjectName(u"calibrateRFBtn")
        self.calibrateRFBtn.setMinimumSize(QSize(0, 30))
        self.calibrateRFBtn.setFont(font1)
        self.calibrateRFBtn.setCheckable(True)

        self.verticalLayout_5.addWidget(self.calibrateRFBtn)

        self.moveBtn = QPushButton(self.centralwidget)
        self.moveBtn.setObjectName(u"moveBtn")
        self.moveBtn.setMinimumSize(QSize(0, 30))
        self.moveBtn.setFont(font1)
        self.moveBtn.setCheckable(True)

        self.verticalLayout_5.addWidget(self.moveBtn)

        self.scanBtn = QPushButton(self.centralwidget)
        self.scanBtn.setObjectName(u"scanBtn")
        self.scanBtn.setMinimumSize(QSize(0, 30))
        self.scanBtn.setFont(font1)
        self.scanBtn.setCheckable(True)

        self.verticalLayout_5.addWidget(self.scanBtn)

        self.RFSlider = QSlider(self.centralwidget)
        self.RFSlider.setObjectName(u"RFSlider")
        sizePolicy5.setHeightForWidth(self.RFSlider.sizePolicy().hasHeightForWidth())
        self.RFSlider.setSizePolicy(sizePolicy5)
        self.RFSlider.setMinimum(-1000)
        self.RFSlider.setMaximum(1000)
        self.RFSlider.setSingleStep(1)
        self.RFSlider.setValue(0)
        self.RFSlider.setOrientation(Qt.Horizontal)
        self.RFSlider.setTickPosition(QSlider.TicksBelow)
        self.RFSlider.setTickInterval(100)

        self.verticalLayout_5.addWidget(self.RFSlider)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer)


        self.horizontalLayout_3.addLayout(self.verticalLayout_5)


        self.mainContentLayout.addLayout(self.horizontalLayout_3)

        self.MLDataBtn = QPushButton(self.centralwidget)
        self.MLDataBtn.setObjectName(u"MLDataBtn")
        self.MLDataBtn.setMinimumSize(QSize(0, 30))
        self.MLDataBtn.setFont(font1)
        self.MLDataBtn.setCheckable(True)

        self.mainContentLayout.addWidget(self.MLDataBtn)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.mainContentLayout.addItem(self.verticalSpacer_2)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.displayBox = QPlainTextEdit(self.centralwidget)
        self.displayBox.setObjectName(u"displayBox")
        sizePolicy7 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.displayBox.sizePolicy().hasHeightForWidth())
        self.displayBox.setSizePolicy(sizePolicy7)
        font4 = QFont()
        font4.setPointSize(10)
        font4.setBold(True)
        font4.setWeight(75)
        self.displayBox.setFont(font4)
        self.displayBox.setFrameShape(QFrame.StyledPanel)
        self.displayBox.setFrameShadow(QFrame.Sunken)
        self.displayBox.setLineWidth(2)
        self.displayBox.setCenterOnScroll(False)

        self.verticalLayout.addWidget(self.displayBox)


        self.mainContentLayout.addLayout(self.verticalLayout)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.stopBtn = QPushButton(self.centralwidget)
        self.stopBtn.setObjectName(u"stopBtn")
        sizePolicy6.setHeightForWidth(self.stopBtn.sizePolicy().hasHeightForWidth())
        self.stopBtn.setSizePolicy(sizePolicy6)
        self.stopBtn.setMinimumSize(QSize(300, 50))
        self.stopBtn.setFont(font4)

        self.verticalLayout_2.addWidget(self.stopBtn)

        self.quitBtn = QPushButton(self.centralwidget)
        self.quitBtn.setObjectName(u"quitBtn")
        sizePolicy6.setHeightForWidth(self.quitBtn.sizePolicy().hasHeightForWidth())
        self.quitBtn.setSizePolicy(sizePolicy6)
        self.quitBtn.setMinimumSize(QSize(300, 50))
        self.quitBtn.setFont(font4)

        self.verticalLayout_2.addWidget(self.quitBtn)


        self.mainContentLayout.addLayout(self.verticalLayout_2)


        self.mainLayout.addLayout(self.mainContentLayout)

        self.SHViewer = SHViewer(self.centralwidget)
        self.SHViewer.setObjectName(u"SHViewer")
        sizePolicy3.setHeightForWidth(self.SHViewer.sizePolicy().hasHeightForWidth())
        self.SHViewer.setSizePolicy(sizePolicy3)
        self.SHViewer.setMinimumSize(QSize(1046, 1046))

        self.mainLayout.addWidget(self.SHViewer)


        self.horizontalLayout.addLayout(self.mainLayout)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"SensorbasedAO", None))
        self.initialiseBtn.setText(QCoreApplication.translate("MainWindow", u"Initialise SB", None))
        self.positionBtn.setText(QCoreApplication.translate("MainWindow", u"Position SB", None))
        self.centroidBtn.setText(QCoreApplication.translate("MainWindow", u"Calibrate-Sys", None))
        self.calibrateBtn.setText(QCoreApplication.translate("MainWindow", u"Calibrate-S", None))
        self.conversionBtn.setText(QCoreApplication.translate("MainWindow", u"S-Z Conv", None))
        self.calibrateBtn_2.setText(QCoreApplication.translate("MainWindow", u"Calibrate-Z", None))
        self.ZernikeArrEdt.setText("")
        self.ZernikeArrEdt.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Separate coefficients with space", None))
        self.ZernikeLbl.setText(QCoreApplication.translate("MainWindow", u"Zernike coefficients (microns)", None))
        self.ZernikeOKBtn.setText(QCoreApplication.translate("MainWindow", u"OK", None))
        self.ZernikeAOBtn_3.setText(QCoreApplication.translate("MainWindow", u"Zernike AO 3", None))
        self.slopeAOBtn_3.setText(QCoreApplication.translate("MainWindow", u"Slope AO 3", None))
        self.ZernikeAOBtn_2.setText(QCoreApplication.translate("MainWindow", u"Zernike AO 2", None))
        self.slopeAOBtn_2.setText(QCoreApplication.translate("MainWindow", u"Slope AO 2", None))
        self.ZernikeAOBtn_1.setText(QCoreApplication.translate("MainWindow", u"Zernike AO 1", None))
        self.slopeAOBtn_1.setText(QCoreApplication.translate("MainWindow", u"Slope AO 1", None))
        self.ZernikeTestBtn.setText(QCoreApplication.translate("MainWindow", u"Zernike Test", None))
        self.ZernikeFullBtn.setText(QCoreApplication.translate("MainWindow", u"Zernike Full", None))
        self.slopeFullBtn.setText(QCoreApplication.translate("MainWindow", u"Slope Full", None))
        self.imageAcqLbl.setText(QCoreApplication.translate("MainWindow", u"Image Acquisition", None))
        self.liveAcqBtn.setText(QCoreApplication.translate("MainWindow", u"Live Acq", None))
        self.burstAcqBtn.setText(QCoreApplication.translate("MainWindow", u"Burst Acq", None))
        self.singleAcqBtn.setText(QCoreApplication.translate("MainWindow", u"Single Acq", None))
        self.miscellaneousLbl.setText(QCoreApplication.translate("MainWindow", u"Miscellaneous", None))
        self.DMRstBtn.setText(QCoreApplication.translate("MainWindow", u"Reset DM", None))
        self.scannerRstBtn.setText(QCoreApplication.translate("MainWindow", u"Reset Scanner", None))
        self.remoteFocusLbl.setText(QCoreApplication.translate("MainWindow", u"Remote Focusing Unit", None))
        self.scanFocusCheck.setText(QCoreApplication.translate("MainWindow", u"Scan Focus", None))
        self.focusDepthLbl.setText(QCoreApplication.translate("MainWindow", u"Focus Depth (microns)", None))
        self.stepIncreLbl.setText(QCoreApplication.translate("MainWindow", u"Step Increment (microns)", None))
        self.stepNumLbl.setText(QCoreApplication.translate("MainWindow", u"Step Number", None))
        self.startDepthLbl.setText(QCoreApplication.translate("MainWindow", u"Start Depth (microns)", None))
        self.pauseTimeLbl.setText(QCoreApplication.translate("MainWindow", u"Depth Pause Time (s)", None))
        self.AOTypeLbl.setText(QCoreApplication.translate("MainWindow", u"AO Correction Type", None))
        self.AOTypeCombo.setItemText(0, QCoreApplication.translate("MainWindow", u"None", None))
        self.AOTypeCombo.setItemText(1, QCoreApplication.translate("MainWindow", u"Zernike AO 3", None))
        self.AOTypeCombo.setItemText(2, QCoreApplication.translate("MainWindow", u"Zernike Full", None))
        self.AOTypeCombo.setItemText(3, QCoreApplication.translate("MainWindow", u"Slope AO 3", None))
        self.AOTypeCombo.setItemText(4, QCoreApplication.translate("MainWindow", u"Slope Full", None))

        self.calibrateRFBtn.setText(QCoreApplication.translate("MainWindow", u"CALIBRATE", None))
        self.moveBtn.setText(QCoreApplication.translate("MainWindow", u"MOVE", None))
        self.scanBtn.setText(QCoreApplication.translate("MainWindow", u"SCAN", None))
        self.MLDataBtn.setText(QCoreApplication.translate("MainWindow", u"ML Dataset", None))
        self.displayBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Sensorbased AO Software", None))
        self.stopBtn.setText(QCoreApplication.translate("MainWindow", u"STOP", None))
        self.quitBtn.setText(QCoreApplication.translate("MainWindow", u"QUIT", None))
    # retranslateUi

