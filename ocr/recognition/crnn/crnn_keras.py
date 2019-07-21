# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import numpy as np
import tensorflow as tf

from . import keys
from .network_keras import keras_crnn as CRNN
from .utils import strLabelConverter, resizeNormalize
from ...config import ocrModelKeras

graph = tf.get_default_graph()  # 解决web.py 相关报错问题


def crnnSource():
    alphabet = keys.alphabetChinese  # 中英文模型
    converter = strLabelConverter(alphabet)
    model = CRNN(32, 1, len(alphabet) + 1, 256, 1, lstmFlag=False)
    model.load_weights(ocrModelKeras)
    return model, converter


# 加载模型
model, converter = crnnSource()


def crnnOcr(image):
    """
    crnn模型，ocr识别
    image:PIL.Image.convert("L")
    """
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    transformer = resizeNormalize((w, 32))
    image = transformer(image)
    image = image.astype(np.float32)
    image = np.array([[image]])
    global graph
    with graph.as_default():
        preds = model.predict(image)
    preds = preds[0]
    preds = np.argmax(preds, axis=2).reshape((-1,))
    sim_pred = converter.decode(preds)
    return sim_pred
