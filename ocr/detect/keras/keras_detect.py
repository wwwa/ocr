# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K

from ocr.config import IMGSIZE, keras_anchors, class_names, kerasTextModel
from .keras_yolo3 import yolo_text, box_layer
from ..detector.detectors import TextDetector
from ..utils.image import get_boxes, letterbox_image
from ..utils.image import resize_im, draw_boxes

graph = tf.get_default_graph()
sess = K.get_session()

anchors = [float(x) for x in keras_anchors.split(',')]
anchors = np.array(anchors).reshape(-1, 2)
num_anchors = len(anchors)

num_classes = len(class_names)
textModel = yolo_text(num_classes, anchors)
textModel.load_weights(kerasTextModel)

image_shape = K.placeholder(shape=(2,))  # 图像原尺寸:h,w
input_shape = K.placeholder(shape=(2,))  # 图像resize尺寸:h,w
box_score = box_layer([*textModel.output, image_shape, input_shape], anchors, num_classes)


def text_detect(img, prob=0.3):
    im = Image.fromarray(img)
    scale = IMGSIZE[0]
    w, h = im.size
    w_, h_ = resize_im(w, h, scale=scale, max_scale=2048)  # 短边固定为608,长边max_scale<4000
    boxed_image, f = letterbox_image(im, (w_, h_))
    boxed_image = im.resize((w_, h_), Image.BICUBIC)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    imgShape = np.array([[h, w]])
    inputShape = np.array([[h_, w_]])

    global graph
    with graph.as_default():
        # 定义 graph变量 解决web.py 相关报错问题
        """
        pred = textModel.predict_on_batch([image_data,imgShape,inputShape])
        box,scores = pred[:,:4],pred[:,-1]

        """
        sess.run(tf.global_variables_initializer())
        box, scores = sess.run(
            [box_score],
            feed_dict={
                textModel.input: image_data,
                input_shape: [h_, w_],
                image_shape: [h, w],
                K.learning_phase(): 0
            })[0]
    keep = np.where(scores > prob)
    box[:, 0:4][box[:, 0:4] < 0] = 0
    box[:, 0][box[:, 0] >= w] = w - 1
    box[:, 1][box[:, 1] >= h] = h - 1
    box[:, 2][box[:, 2] >= w] = w - 1
    box[:, 3][box[:, 3] >= h] = h - 1
    box = box[keep[0]]
    scores = scores[keep[0]]
    return box, scores


def keras_detect(img,
                 MAX_HORIZONTAL_GAP=30,
                 MIN_V_OVERLAPS=0.6,
                 MIN_SIZE_SIM=0.6,
                 TEXT_PROPOSALS_MIN_SCORE=0.7,
                 TEXT_PROPOSALS_NMS_THRESH=0.3,
                 TEXT_LINE_NMS_THRESH=0.3,
                 ):
    boxes, scores = text_detect(np.array(img))
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    textdetector = TextDetector(MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)
    shape = img.shape[:2]
    boxes = textdetector.detect(boxes,
                                scores[:, np.newaxis],
                                shape,
                                TEXT_PROPOSALS_MIN_SCORE,
                                TEXT_PROPOSALS_NMS_THRESH,
                                TEXT_LINE_NMS_THRESH,
                                )

    text_recs = get_boxes(boxes)
    newBox = []
    rx = 1
    ry = 1
    for box in text_recs:
        x1, y1 = (box[0], box[1])
        x2, y2 = (box[2], box[3])
        x3, y3 = (box[6], box[7])
        x4, y4 = (box[4], box[5])
        newBox.append([x1 * rx, y1 * ry, x2 * rx, y2 * ry, x3 * rx, y3 * ry, x4 * rx, y4 * ry])

    img_drawed = draw_boxes(img, newBox)

    return newBox, img_drawed
