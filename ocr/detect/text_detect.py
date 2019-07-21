# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import os
import cv2
import numpy as np

from .detector.detectors import TextDetector
from .utils.image import get_boxes, draw_boxes

MAX_HORIZONTAL_GAP = 30
MIN_V_OVERLAPS = 0.6
MIN_SIZE_SIM = 0.6
TEXT_PROPOSALS_MIN_SCORE = 0.7
TEXT_PROPOSALS_NMS_THRESH = 0.3
TEXT_LINE_NMS_THRESH = 0.3


def text_detect(img, detect_func, output_path):
    boxes, scores = detect_func(img)
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
    draw_boxes(img, boxes)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    # cv2.imwrite(os.path.join(output_path, 'test'), img)

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
    return newBox
