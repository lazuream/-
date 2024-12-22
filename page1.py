# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'page1.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1266, 727)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(20, 20, 1231, 701))
        self.widget.setStyleSheet("border-radius:10px;\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0.322034 rgba(97, 214, 239, 255), stop:0.836158 rgba(213, 165, 248, 255));")
        self.widget.setObjectName("widget")
        self.frame = QtWidgets.QFrame(self.widget)
        self.frame.setGeometry(QtCore.QRect(20, 50, 181, 601))
        self.frame.setStyleSheet("background-color: rgb(161, 255, 249);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.pushButton_Geometric = QtWidgets.QPushButton(self.frame)
        self.pushButton_Geometric.setGeometry(QtCore.QRect(30, 30, 121, 31))
        self.pushButton_Geometric.setStyleSheet("QPushButton {\n"
"                            border: 2px solid black;\n"
"                            border-radius: 10px; /* 圆角半径 */\n"
"                            padding: 6px; /* 按钮内边距 */\n"
"                            background-color: white;\n"
"                            min-width: 80px;\n"
"                        }\n"
"                        QPushButton:hover {\n"
"                            background-color: lightgray;\n"
"                        }\n"
"                        QPushButton:pressed {\n"
"                            background-color: gray;\n"
"                            padding-top: 8px;\n"
"                            padding-left: 8px;\n"
"                            padding-bottom: 4px;\n"
"                            padding-right: 4px;\n"
"                        }")
        self.pushButton_Geometric.setObjectName("pushButton_Geometric")
        self.pushButton_Contrast = QtWidgets.QPushButton(self.frame)
        self.pushButton_Contrast.setGeometry(QtCore.QRect(30, 80, 121, 31))
        self.pushButton_Contrast.setStyleSheet("QPushButton {\n"
"                            border: 2px solid black;\n"
"                            border-radius: 10px; /* 圆角半径 */\n"
"                            padding: 6px; /* 按钮内边距 */\n"
"                            background-color: white;\n"
"                            min-width: 80px;\n"
"                        }\n"
"                        QPushButton:hover {\n"
"                            background-color: lightgray;\n"
"                        }\n"
"                        QPushButton:pressed {\n"
"                            background-color: gray;\n"
"                            padding-top: 8px;\n"
"                            padding-left: 8px;\n"
"                            padding-bottom: 4px;\n"
"                            padding-right: 4px;\n"
"                        }")
        self.pushButton_Contrast.setObjectName("pushButton_Contrast")
        self.pushButton_Smooth = QtWidgets.QPushButton(self.frame)
        self.pushButton_Smooth.setGeometry(QtCore.QRect(30, 130, 121, 31))
        self.pushButton_Smooth.setStyleSheet("QPushButton {\n"
"                            border: 2px solid black;\n"
"                            border-radius: 10px; /* 圆角半径 */\n"
"                            padding: 6px; /* 按钮内边距 */\n"
"                            background-color: white;\n"
"                            min-width: 80px;\n"
"                        }\n"
"                        QPushButton:hover {\n"
"                            background-color: lightgray;\n"
"                        }\n"
"                        QPushButton:pressed {\n"
"                            background-color: gray;\n"
"                            padding-top: 8px;\n"
"                            padding-left: 8px;\n"
"                            padding-bottom: 4px;\n"
"                            padding-right: 4px;\n"
"                        }")
        self.pushButton_Smooth.setObjectName("pushButton_Smooth")
        self.pushButton_Partition = QtWidgets.QPushButton(self.frame)
        self.pushButton_Partition.setGeometry(QtCore.QRect(30, 180, 121, 31))
        self.pushButton_Partition.setStyleSheet("QPushButton {\n"
"                            border: 2px solid black;\n"
"                            border-radius: 10px; /* 圆角半径 */\n"
"                            padding: 6px; /* 按钮内边距 */\n"
"                            background-color: white;\n"
"                            min-width: 80px;\n"
"                        }\n"
"                        QPushButton:hover {\n"
"                            background-color: lightgray;\n"
"                        }\n"
"                        QPushButton:pressed {\n"
"                            background-color: gray;\n"
"                            padding-top: 8px;\n"
"                            padding-left: 8px;\n"
"                            padding-bottom: 4px;\n"
"                            padding-right: 4px;\n"
"                        }")
        self.pushButton_Partition.setObjectName("pushButton_Partition")
        self.pushButton_add = QtWidgets.QPushButton(self.frame)
        self.pushButton_add.setGeometry(QtCore.QRect(40, 490, 96, 31))
        self.pushButton_add.setStyleSheet("QPushButton {\n"
"                            border: 2px solid black;\n"
"                            border-radius: 10px; /* 圆角半径 */\n"
"                            padding: 6px; /* 按钮内边距 */\n"
"                            background-color: white;\n"
"                            min-width: 80px;\n"
"                        }\n"
"                        QPushButton:hover {\n"
"                            background-color: lightgray;\n"
"                        }\n"
"                        QPushButton:pressed {\n"
"                            background-color: gray;\n"
"                            padding-top: 8px;\n"
"                            padding-left: 8px;\n"
"                            padding-bottom: 4px;\n"
"                            padding-right: 4px;\n"
"                        }")
        self.pushButton_add.setObjectName("pushButton_add")
        self.pushButton_save = QtWidgets.QPushButton(self.frame)
        self.pushButton_save.setGeometry(QtCore.QRect(40, 550, 96, 31))
        self.pushButton_save.setStyleSheet("QPushButton {\n"
"                            border: 2px solid black;\n"
"                            border-radius: 10px; /* 圆角半径 */\n"
"                            padding: 6px; /* 按钮内边距 */\n"
"                            background-color: white;\n"
"                            min-width: 80px;\n"
"                        }\n"
"                        QPushButton:hover {\n"
"                            background-color: lightgray;\n"
"                        }\n"
"                        QPushButton:pressed {\n"
"                            background-color: gray;\n"
"                            padding-top: 8px;\n"
"                            padding-left: 8px;\n"
"                            padding-bottom: 4px;\n"
"                            padding-right: 4px;\n"
"                        }")
        self.pushButton_save.setObjectName("pushButton_save")
        self.frame_2 = QtWidgets.QFrame(self.widget)
        self.frame_2.setGeometry(QtCore.QRect(210, 50, 981, 601))
        self.frame_2.setStyleSheet("background-color: rgb(249, 255, 220);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.frame_initialimage = QtWidgets.QFrame(self.frame_2)
        self.frame_initialimage.setGeometry(QtCore.QRect(30, 40, 441, 501))
        self.frame_initialimage.setStyleSheet("background-color: rgb(59, 59, 89);")
        self.frame_initialimage.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_initialimage.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_initialimage.setObjectName("frame_initialimage")
        self.frame_retimage = QtWidgets.QFrame(self.frame_2)
        self.frame_retimage.setGeometry(QtCore.QRect(510, 40, 441, 501))
        self.frame_retimage.setStyleSheet("background-color: rgb(59, 59, 89);")
        self.frame_retimage.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_retimage.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_retimage.setObjectName("frame_retimage")
        self.scrollArea_retimage = QtWidgets.QScrollArea(self.frame_retimage)
        self.scrollArea_retimage.setGeometry(QtCore.QRect(0, 0, 461, 521))
        self.scrollArea_retimage.setStyleSheet("border-radius:10px;")
        self.scrollArea_retimage.setWidgetResizable(False)
        self.scrollArea_retimage.setObjectName("scrollArea_retimage")
        self.scrollAreaWidgetContents_retimage = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_retimage.setGeometry(QtCore.QRect(0, 0, 4000, 4000))
        self.scrollAreaWidgetContents_retimage.setStyleSheet("border-radius:10px;")
        self.scrollAreaWidgetContents_retimage.setObjectName("scrollAreaWidgetContents_retimage")
        self.scrollArea_retimage.setWidget(self.scrollAreaWidgetContents_retimage)
        self.textEdit = QtWidgets.QTextEdit(self.frame_2)
        self.textEdit.setGeometry(QtCore.QRect(30, 560, 431, 31))
        self.textEdit.setStyleSheet("background-color: rgb(249, 255, 220);")
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(self.frame_2)
        self.textEdit_2.setGeometry(QtCore.QRect(510, 560, 441, 31))
        self.textEdit_2.setStyleSheet("background-color: rgb(249, 255, 220);")
        self.textEdit_2.setOverwriteMode(True)
        self.textEdit_2.setObjectName("textEdit_2")
        self.transformationContainer_1 = QtWidgets.QWidget(self.widget)
        self.transformationContainer_1.setGeometry(QtCore.QRect(179, 80, 131, 141))
        self.transformationContainer_1.setToolTipDuration(0)
        self.transformationContainer_1.setStyleSheet("background-color: rgb(226, 255, 255);")
        self.transformationContainer_1.setObjectName("transformationContainer_1")
        self.transformationContainer_2 = QtWidgets.QWidget(self.widget)
        self.transformationContainer_2.setGeometry(QtCore.QRect(179, 130, 131, 141))
        self.transformationContainer_2.setToolTipDuration(0)
        self.transformationContainer_2.setStyleSheet("background-color: rgb(226, 255, 255);")
        self.transformationContainer_2.setObjectName("transformationContainer_2")
        self.transformationContainer_3 = QtWidgets.QWidget(self.widget)
        self.transformationContainer_3.setGeometry(QtCore.QRect(180, 180, 131, 151))
        self.transformationContainer_3.setToolTipDuration(0)
        self.transformationContainer_3.setStyleSheet("background-color: rgb(226, 255, 255);")
        self.transformationContainer_3.setObjectName("transformationContainer_3")
        self.transformationContainer_4 = QtWidgets.QWidget(self.widget)
        self.transformationContainer_4.setGeometry(QtCore.QRect(179, 230, 131, 151))
        self.transformationContainer_4.setToolTipDuration(0)
        self.transformationContainer_4.setStyleSheet("background-color: rgb(226, 255, 255);")
        self.transformationContainer_4.setObjectName("transformationContainer_4")
        self.pushButton_exit = QtWidgets.QPushButton(self.widget)
        self.pushButton_exit.setGeometry(QtCore.QRect(1184, 10, 31, 31))
        self.pushButton_exit.setStyleSheet("#pushButton_exit {\n"
"    background-color: rgb(160, 238, 171);\n"
"}\n"
"\n"
"/* 定义鼠标悬停时的样式 */\n"
"#pushButton_exit:hover {\n"
"    background-color: rgb(221, 84, 66);\n"
"}")
        self.pushButton_exit.setObjectName("pushButton_exit")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_Geometric.setText(_translate("MainWindow", "几何运算"))
        self.pushButton_Contrast.setText(_translate("MainWindow", "对比度增强"))
        self.pushButton_Smooth.setText(_translate("MainWindow", "平滑处理"))
        self.pushButton_Partition.setText(_translate("MainWindow", "图像分割"))
        self.pushButton_add.setText(_translate("MainWindow", "添加图片"))
        self.pushButton_save.setText(_translate("MainWindow", "保存图片"))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">初始图像</p></body></html>"))
        self.textEdit_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">处理后图像</p></body></html>"))
        self.pushButton_exit.setText(_translate("MainWindow", "X"))
