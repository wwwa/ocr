# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import os
import shutil
from glob import glob

import cv2

from ocr import ROOT_PATH
from ocr.detect.ctpn.ctpn_detect import text_detect

output_path = os.path.join(ROOT_PATH, 'output_data', 'ctpn')
input_path = os.path.join(ROOT_PATH, 'data', '268', 'page', '*.jpg')
# input_path = os.path.join(ROOT_PATH, 'data', 'test', 'i*.jpeg')

if os.path.exists(output_path):
    shutil.rmtree(output_path)

if not os.path.exists(output_path):
    os.makedirs(output_path)


def text_box_detect(image_name):
    base_name = image_name.split('/')[-1]
    img = cv2.imread(image_name)
    text_recs, img_drawed = text_detect(img)
    cv2.imwrite(os.path.join(output_path, base_name), img_drawed)


if __name__ == "__main__":
    im_names = glob(input_path)
    for name in sorted(im_names):
        text_box_detect(name)
        break
