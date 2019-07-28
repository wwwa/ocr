# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import traceback

from ocr.config import DETECT_PATH
from .lib.fast_rcnn.config import cfg, cfg_from_file
from .lib.fast_rcnn.test import test_ctpn
from .lib.networks.factory import get_network
from .lib.text_connector.detectors import TextDetector
from .lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale is not None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, boxes, scale):
    box_id = 0
    img = img.copy()
    text_recs = np.zeros((len(boxes), 8), np.int)
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue

        if box[8] >= 0.8:
            color = (255, 0, 0)  # red
        else:
            color = (0, 255, 0)  # green

        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

        for i in range(8):
            text_recs[box_id, i] = box[i]

        box_id += 1

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    return text_recs, img


def load_tf_model():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    # init session
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print('Loading network {:s}... '.format("VGGnet_test"))
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except Exception as e:
        traceback.print_exc()
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
    return sess, net


sess, net = load_tf_model()


def ctpn_detect(img):
    cfg_from_file(os.path.join(DETECT_PATH, 'ctpn', 'text.yml'))

    img, scale = resize_im(np.array(img), scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)
    sess.close()

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    # return boxes
    return scores, boxes, img, scale


def text_detect(img):
    scores, boxes, img, scale = ctpn_detect(img)
    text_recs, img_drawed = draw_boxes(img, boxes, scale)
    return text_recs, img_drawed
