# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import cv2
import numpy as np

from ocr.config import IMGSIZE, yoloCfg, yoloWeights

textNet = cv2.dnn.readNetFromDarknet(yoloCfg, yoloWeights)  # 文字定位


def text_detect(img):
    thresh = 0
    img_height, img_width = img.shape[:2]
    inputBlob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=IMGSIZE, swapRB=True, crop=False)
    textNet.setInput(inputBlob / 255.0)
    outputName = textNet.getUnconnectedOutLayersNames()
    outputs = textNet.forward(outputName)
    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > thresh:
                center_x = int(detection[0] * img_width)
                center_y = int(detection[1] * img_height)
                width = int(detection[2] * img_width)
                height = int(detection[3] * img_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                if class_id == 1:
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, left + width, top + height])

    return np.array(boxes), np.array(confidences)
