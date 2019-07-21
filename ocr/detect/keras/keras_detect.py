# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import numpy as np
from PIL import Image

from ocr.config import IMGSIZE, keras_anchors, class_names, kerasTextModel
from ..utils.image import resize_im

from .keras_yolo3 import yolo_text, box_layer

import tensorflow as tf
from keras import backend as K

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


def text_detect(img, prob=0.05):
    im = Image.fromarray(img)
    scale = IMGSIZE[0]
    w, h = im.size
    w_, h_ = resize_im(w, h, scale=scale, max_scale=2048)  # 短边固定为608,长边max_scale<4000
    # boxed_image,f = letterbox_image(im, (w_,h_))
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
