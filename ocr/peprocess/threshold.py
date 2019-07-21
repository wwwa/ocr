# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import os
import cv2
import matplotlib.pyplot as plt
from ocr import IMAGE_PATH
import imutils


def threshold(img):
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    titles = ['img', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()

    for title, image in zip(titles, images):
        cv2.imshow(title, image)


def adaptive_threshold(img):
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 11, 2)  # 换行符号
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 3, 2)  # 换行符号
    titles = ['img', 'th1', 'th2', 'th3']
    images = [img, th1, th2, th3]
    # plt.figure()
    # for i in range(4):
    #     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    # plt.show()
    for title, image in zip(titles, images):
        cv2.imshow(title, image)


def otsu_threshold(img):
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # (5,5)为高斯核的大小，0为标准差
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 阀值一定要设为0
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    images = [img, 0, th1,
              img, 0, th2,
              img, 0, th3]
    titles = ['original noisy image', 'histogram', 'global thresholding(v=127)',
              'original noisy image', 'histogram', "otsu's thresholding",
              'gaussian giltered image', 'histogram', "otus's thresholding"]
    # 这里使用了pyplot中画直方图的方法，plt.hist要注意的是他的参数是一维数组
    # 所以这里使用了（numpy）ravel方法，将多维数组转换成一维，也可以使用flatten方法
    # for i in range(3):
    #     plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    #     plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    #     plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    #     plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    #
    # plt.show()

    cv2.imshow('th1', th1)
    cv2.imshow('th2', th2)
    cv2.imshow('th3', th3)


if __name__ == "__main__":
    image_path = os.path.join(IMAGE_PATH, "119.jpeg")

    image = cv2.imread(image_path, 0)  # 直接读为灰度图像
    image = imutils.resize(image, height=500)

    # threshold(image)
    # adaptive_threshold(image)
    otsu_threshold(image)

    cv2.waitKey(0)  # 等待按键按下
    cv2.destroyAllWindows()  # 清除所有窗口
