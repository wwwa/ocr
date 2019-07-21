# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""
import os

from ocr import ROOT_PATH

DETECT_PATH = os.path.join(ROOT_PATH, 'models', 'det')
RECOGNITION_PATH = os.path.join(ROOT_PATH, 'models', 'rec')
IMGSIZE = (608, 608)  # yolo3 输入图像尺寸

# keras 版本anchors
keras_anchors = '8,11, 8,16, 8,23, 8,33, 8,48, 8,97, 8,139, 8,198, 8,283'
class_names = ['none', 'text', ]
kerasTextModel = os.path.join(DETECT_PATH, 'keras', "text.h5")  # keras版本模型权重文件

# GPU选择及启动GPU序号
GPU = False  # OCR 是否启用GPU
GPUID = 0  # 调用GPU序号

# nms选择,支持cython,gpu,python
nmsFlag = 'gpu'  # cython/gpu/python 容错性 优先启动GPU，其次是cpython 最后是python
if not GPU:
    nmsFlag = 'cython'

# darknet yolo
darknetRoot = os.path.join(DETECT_PATH, 'darknet', "darknet")  # yolo 安装目录
yoloCfg = os.path.join(DETECT_PATH, 'darknet', "text.cfg")
yoloWeights = os.path.join(DETECT_PATH, 'darknet', "text.weights")
yoloData = os.path.join(DETECT_PATH, 'darknet', "text.data")

# 是否启用LSTM crnn模型
# OCR模型是否调用LSTM层
LSTMFLAG = True
# 模型选择 True:中英文模型 False:英文模型
ocrFlag = 'torch'  # ocr模型 支持 keras  torch版本
# ocrFlag = 'keras'# ocr模型 支持 keras  torch版本
chinsesModel = True
ocrModelKeras = os.path.join(RECOGNITION_PATH, 'keras', "ocr-dense-keras.h5")  # keras版本OCR，暂时支持dense
if chinsesModel:
    if LSTMFLAG:
        ocrModel = os.path.join(RECOGNITION_PATH, 'torch', "ocr-lstm.pth")
    else:
        ocrModel = os.path.join(RECOGNITION_PATH, "ocr-dense.pth")
else:
    ##纯英文模型
    LSTMFLAG = True
    ocrModel = os.path.join(RECOGNITION_PATH, "ocr-english.pth")
