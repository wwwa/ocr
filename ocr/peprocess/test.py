# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import os
import cv2
import numpy as np

from ocr import IMAGE_PATH


def edge_check(img):
    # Canny 边缘检测
    canny_img = cv2.Canny(img, 200, 300)
    cv2.imshow("canny", canny_img)


def contour_check(img):
    # 轮廓检测
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)
    cv2.imshow("contour", color)


def line_check(img):
    # 直线检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    minLineLenght = 20
    maxLineGap = 1
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLenght, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("edges", edges)
    cv2.imshow("lines", img)


if __name__ == "__main__":
    image_path = os.path.join(IMAGE_PATH, "119.jpeg")
    image = cv2.imread(image_path)
    img = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("origin", img)

    # Canny 边缘检测
    # edge_check(img)
    # 轮廓检测
    # contour_check(img)
    # 直线检测
    line_check(img)

    cv2.waitKey()
    cv2.destroyAllWindows()
    import numpy as np
    np.maximum()
    np.where
