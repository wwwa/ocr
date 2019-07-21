# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""
import os
import cv2
from glob import glob

from ocr.detect.keras.keras_detect import text_detect as detect_func
from ocr.detect.text_detect import text_detect

from ocr import ROOT_PATH

output_path = os.path.join(ROOT_PATH, 'output_data', 'keras')
# input_path = os.path.join(ROOT_PATH, 'data', '268', 'page', '*.jpg')
input_path = os.path.join(ROOT_PATH, 'data', 'test', 'i*.jpeg')

if not os.path.exists(output_path):
    os.makedirs(output_path)


def text_box_detect(image_name):
    base_name = image_name.split('/')[-1]
    img = cv2.imread(image_name)
    text_detect(img, detect_func, output_path)


if __name__ == "__main__":
    im_names = glob(input_path)
    for name in sorted(im_names):
        text_box_detect(name)
        break
