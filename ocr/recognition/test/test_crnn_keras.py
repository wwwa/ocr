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
import numpy as np
from PIL import Image

from ocr import ROOT_PATH
from ocr.detect.ctpn.ctpn_detect import text_detect
from ocr.recognition.crnn.crnn_keras import crnnOcr
from ocr.utils.image import solve, rotate_cut_img, sort_box

output_path = os.path.join(ROOT_PATH, 'output_data', 'ctpn')
input_path = os.path.join(ROOT_PATH, 'data', '268', 'page', '*.jpg')
# input_path = os.path.join(ROOT_PATH, 'data', 'test', 'i*.jpeg')

if os.path.exists(output_path):
    shutil.rmtree(output_path)

os.makedirs(output_path)


def crnn_rec(im, boxes, left_adjust=False, right_adjust=False, alph=0.2, f=1.0):
    """
    crnn模型，ocr识别
    leftAdjust,rightAdjust 是否左右调整box 边界误差，解决文字漏检
    """
    results = []
    im = Image.fromarray(im)
    for index, box in enumerate(boxes):
        degree, w, h, cx, cy = solve(box)
        part_img, new_w, new_h = rotate_cut_img(im, degree, box, w, h, left_adjust, right_adjust, alph)
        text = crnnOcr(part_img.convert('L'))
        if text.strip() != u'':
            results.append({'cx': cx * f, 'cy': cy * f, 'text': text, 'w': new_w * f, 'h': new_h * f,
                            'degree': degree * 180.0 / np.pi})

    return results


def text_recognition(image_name, alph=0.01):
    left_adjust = True
    right_adjust = True
    img = cv2.imread(image_name)
    # img, f = letterbox_image(Image.fromarray(img), IMGSIZE)  # pad
    text_recs, img_drawed = text_detect(img)  # 文字检测
    newBox = sort_box(text_recs)  # 行文本识别
    result = crnn_rec(np.array(img), newBox, left_adjust, right_adjust, alph, 1.0)
    return result


if __name__ == "__main__":
    im_names = glob(input_path)
    for name in sorted(im_names):
        result = text_recognition(name)
        break
    # import tensorflow as tf
    # from ocr.detect.ctpn.ctpn_detect import get_network
    # from tensorflow.python import pywrap_tensorflow
    #
    # net = get_network("VGGnet_test")
    # md, ckpt = [], []
    # for key in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    #     # print(key.name)
    #     md.append(key.name)
    # # print("*" * 100)
    # checkpoint_path = os.path.join(ROOT_PATH, "models", 'det', 'ctpn', 'checkpoints', 'VGGnet_fast_rcnn_iter_50000.ckpt')
    # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in sorted(var_to_shape_map):
    #     # print("tensor_name: ", key)
    #     # print(reader.get_tensor(key))
    #     # print(key)
    #     ckpt.append(key)
    # md = sorted(md)
    # ckpt = sorted(ckpt)
    # print('\n'.join(md))
    # print('*' * 100)
    # print('\n'.join(ckpt))
