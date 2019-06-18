# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import os
import cv2
import numpy as np
import imutils
from ocr import IMAGE_PATH


def find_contours(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    print("*" * 100)
    print(hierarchy)
    # 绘制独立轮廓，如第四个轮廓
    # imag = cv2.drawContour(img,contours,-1,(0,255,0),3)
    # 但是大多数时候，下面方法更有用
    imag = cv2.drawContours(img, contours, 3, (0, 255, 0), 3)
    while (1):
        cv2.imshow('img', img)
        cv2.imshow('imgray', im_gray)
        cv2.imshow('image', image)
        cv2.imshow('imag', imag)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = os.path.join(IMAGE_PATH, "119.jpeg")

    image = cv2.imread(image_path)
    image = imutils.resize(image, height=500)

    find_contours(image)

    cv2.waitKey(0)  # 等待按键按下
    cv2.destroyAllWindows()  # 清除所有窗口
