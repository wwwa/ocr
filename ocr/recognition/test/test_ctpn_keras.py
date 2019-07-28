# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import os
from glob import glob

import cv2
import tensorflow as tf

from ocr import ROOT_PATH
from ocr.recognition.ctpn.rec_keras import rec_keras

output_path = os.path.join(ROOT_PATH, 'output_data', 'ctpn')
input_path = os.path.join(ROOT_PATH, 'data', '268', 'page', '*.jpg')
# input_path = os.path.join(ROOT_PATH, 'data', 'test', 'i*.jpeg')


if __name__ == "__main__":
    print('\n'.join(sorted([key.name for key in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])))
    im_names = glob(input_path)
    for name in sorted(im_names):
        result = rec_keras(cv2.imread(name))
        print(result)
        break
