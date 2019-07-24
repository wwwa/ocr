#!/usr/bin/python
# encoding: utf-8
from PIL import Image
import numpy as np


class strLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + 'ç'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def decode(self, res):
        N = len(res)
        raw = []
        for i in range(N):
            if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
                raw.append(self.alphabet[res[i] - 1])
        return ''.join(raw)


class ResizeNormalize(object):
    """重置图像大小， width(None) x height(32)，并是每个像素点的取值范围变为【-1，1】
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        size = self.size
        img_w, img_h = size
        scale = img.size[1] * 1.0 / img_h
        w = img.size[0] / scale
        w = int(w)
        img = img.resize((w, img_h), self.interpolation)
        # w, h = img.size
        # 使像素范围变为【-1， 1】
        img = (np.array(img) / 255.0 - 0.5) / 0.5
        return img
