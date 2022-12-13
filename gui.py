# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitled.ui'
##
## Created by: Qt User Interface Compiler version 5.15.6
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################
import sys
import threading

import numpy as np
import pydub
import soundfile
import torch
from PySide2 import QtWidgets
from PySide2.QtCore import *  # type: ignore
from PySide2.QtGui import *  # type: ignore
from PySide2.QtWidgets import *  # type: ignore
from tensorflow.python.distribute.multi_process_lib import multiprocessing

import utils
from models import SynthesizerTrn
from text import text_to_phones, cleaned_text_to_sequence, symbols
from pydub.playback import play

hps = utils.get_hparams_from_file("configs/ms.json")
net_g = SynthesizerTrn(
  len(symbols),
  hps.data.filter_length // 2 + 1,
  hps.data.hop_length,
  hps.data.sampling_rate,
  hps.train.segment_size // hps.data.hop_length,
  n_speakers=hps.data.n_speakers,
  **hps.model)

_ = net_g.eval()

_ = utils.load_checkpoint("/Volumes/Extend/下载/G_4800.pth", net_g, None)



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(555, 375)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.textEdit = QTextEdit(self.centralwidget)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(70, 30, 241, 79))
        self.pushButton_3 = QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(340, 30, 113, 32))
        self.pushButton_4 = QPushButton(self.centralwidget)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(340, 70, 113, 32))
        self.scrollArea_2 = ControlArea(self.centralwidget, 60, 150, 300,450)
        self.pushButton_3.clicked.connect(self.infer)
        # self.pushButton_4.clicked.connect(self.clear)
        self.textEdit.textChanged.connect(self.clear)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 555, 24))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    def clear(self):
        self.scrollArea_2.clear_elements()


    def infer(self):
        threshold = 260
        pitch_control = self.scrollArea_2.get_elements_values()
        self.scrollArea_2.clear_elements()
        if len(pitch_control) >0:
            pitch_control = torch.FloatTensor(pitch_control).unsqueeze(0)
            pitch_control[pitch_control<threshold+1] = 0
        else:
            pitch_control = None
        text = self.textEdit.toPlainText()
        phones = text_to_phones(text)
        print(phones)
        text_norm = torch.LongTensor(cleaned_text_to_sequence(phones))
        with torch.no_grad():
            x_tst = text_norm.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([text_norm.size(0)])
            spk = torch.LongTensor([1])
            result = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, sid=spk,
                                length_scale=1,pitch_control=pitch_control)
            audio = result[0][0, 0].data.float().numpy()
            soundfile.write("samples/out.wav", audio, 44100)
            pred_f0 = result[-1].data.float().numpy()
            # data = np.int8(audio * 2 ** 7)
            song = pydub.AudioSegment(np.pad(audio, [0, 20000]).tobytes(),
                                      frame_rate=44100,
                                      sample_width=audio.dtype.itemsize,
                                      channels=1)
            elements = [(ch, pred_f0[0, idx]) for idx, ch in enumerate(phones)]
            pred_f0[pred_f0<threshold] = threshold
            self.scrollArea_2.add_elements(elements,pred_f0[0,:].min()-10,pred_f0[0,:].max()+10)
            sing_process = threading.Thread(target=play, args=(song,))
            sing_process.start()


    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"合成", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"清除", None))
        # self.label.setText(QCoreApplication.translate("MainWindow", u"\u4f60", None))
        # self.label_2.setText(QCoreApplication.translate("MainWindow", "好", None))
    # retranslateUi
    def test_bind(self):
        print("clicked")

class ControlArea(QScrollArea):

    def __init__(self, parent, left ,top, value_min, value_max):
        super().__init__(parent)
        self.slider_width = 21
        self.slider_height = 81

        self.label_width = 27
        self.label_height = 16

        self.elememt_start = 10
        self.elememt_sep = 35

        self.outer_width = 800
        self.outer_height = 141
        self.content_width = 1000


        self.current_start = self.elememt_start

        self.elements = []

        # self.setObjectName(u"scrollArea_2")
        self.setGeometry(QRect(left, top, self.outer_width, self.outer_height))
        self.setWidgetResizable(True)
        self.contents = QWidget()
        self.contents.setObjectName(u"scrollAreaWidgetContents_2")
        self.contents.setGeometry(QRect(0, 0, self.content_width, self.outer_height-2))
        self.contents.setMinimumSize(QSize(self.content_width, self.outer_height-2))
        self.setWidget(self.contents)
        # for i in range(5):
        #     self.add_element(i)

    def add_elements(self, elements, min, max):
        for txt, value in elements:
            self._add_element(txt, value, min, max)

    def get_elements_values(self):
        return [i.get_value() for i in self.elements]

    def _add_element(self, txt, value, min, max):
        # print("add")
        sliderWidget = self.add_slider(self.current_start, 30)
        label_offset = 3
        textWidget = self.add_label(txt, self.current_start+label_offset, 10)
        valueWidget = self.add_label(value, self.current_start+label_offset, 110)
        self.current_start += self.elememt_sep
        element = Element(sliderWidget, textWidget, valueWidget, len(self.elements), min, max)
        element.set_value(value)
        element.show()
        self.elements.append(element)


    def add_slider(self, left: int, top: int):
        verticalSlider = QSlider(self.contents)
        # verticalSlider.setObjectName(f"verticalSlider_{i}")
        verticalSlider.setGeometry(QRect(left, top, self.slider_width, self.slider_height))
        verticalSlider.setOrientation(Qt.Vertical)
        return verticalSlider

    def add_label(self,text, left ,top):
        label = QLabel(self.contents)
        label.setGeometry(QRect(left, top, self.label_width, self.label_height))

        label.setText(QCoreApplication.translate("MainWindow", f"{text}", None))
        return label

    def clear_elements(self):
        [c.deleteLater() for c in self.contents.children()]
        self.elements.clear()
        self.current_start = self.elememt_start


class Element:
    def __init__(self, slider, text, value, name, min, max):
        self.sliderWidget = slider
        self.textWidget = text
        self.valueWidget = value
        self.name = name
        self.min = min
        self.max = max
        self.value = (self.max-self.min)*(self.sliderWidget.value()/99)+self.min
        slider.valueChanged.connect(self.slider_value_change)

    def slider_value_change(self):
        # print(self.sliderWidget.value())
        self.value = (self.max-self.min)*(self.sliderWidget.value()/99)+self.min
        self.valueWidget.setText(QCoreApplication.translate("MainWindow", '{}'.format(round(self.value, 2)), None))

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.valueWidget.setText(QCoreApplication.translate("MainWindow", f"{self.value}", None))
        self.sliderWidget.setValue((value-self.min)*99//(self.max-self.min))

    def show(self):
        self.sliderWidget.show()
        self.textWidget.show()
        self.valueWidget.show()

class UiMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        # 继承父类init方法
        super().__init__()
        self.start_x = None
        self.start_y = None
        self.move_window = True
        self.setupUi(self)
        self.show()


    # 输出日志
    def printf(self, text):
        print(str(text))
        self.text_res.append(str(text))

if __name__ == "__main__":
    app = QApplication()
    win = UiMainWindow()
    win.show()
    sys.exit(app.exec_())