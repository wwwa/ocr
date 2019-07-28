# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""
import cv2
import numpy as np
from PIL import Image


def resize_im(w, h, scale=416, max_scale=608):
    f = float(scale) / min(h, w)
    if max_scale is not None:
        if f * max(h, w) > max_scale:
            f = float(max_scale) / max(h, w)
    new_w, new_h = int(w * f), int(h * f)

    return new_w - (new_w % 32), new_h - (new_h % 32)


def draw_boxes(img, boxes):
    img = img.copy()
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue
        if box[8] >= 0.9:
            color = (0, 0, 255)  # red
        else:
            color = (0, 255, 0)  # green
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
    return img


def get_boxes(bboxes):
    """boxes: bounding boxes
    """
    text_recs = np.zeros((len(bboxes), 8), np.int)
    index = 0
    for box in bboxes:

        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[0]
        y3 = box[5] * box[0] + b2
        x4 = box[2]
        y4 = box[5] * box[2] + b2

        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)
        fTmp0 = y3 - y1
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)
        y = np.fabs(fTmp1 * disY / width)
        if box[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y

        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
        index = index + 1

    return text_recs


def letterbox_image(image, size, fillValue=[128, 128, 128]):
    '''resize image with unchanged aspect ratio using padding'''
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w * 1.0 / image_w, h * 1.0 / image_h))
    new_h = int(image_h * min(w * 1.0 / image_w, h * 1.0 / image_h))

    resized_image = image.resize((new_w, new_h), Image.BICUBIC)
    if fillValue is None:
        fillValue = [int(x.mean()) for x in cv2.split(np.array(image))]
    boxed_image = Image.new('RGB', size, tuple(fillValue))
    boxed_image.paste(resized_image, (0, 0))
    return boxed_image, new_w / image_w


def eval_angle(im, detectAngle=False):
    """
    估计图片偏移角度
    @@param:im
    @@param:detectAngle 是否检测文字朝向
    """
    angle = 0
    img = np.array(im)
    if detectAngle:
        angle = angle_detect(img=np.copy(img))  ##文字朝向检测
        if angle == 90:
            im = Image.fromarray(im).transpose(Image.ROTATE_90)
        elif angle == 180:
            im = Image.fromarray(im).transpose(Image.ROTATE_180)
        elif angle == 270:
            im = Image.fromarray(im).transpose(Image.ROTATE_270)
        img = np.array(im)

    return angle, img


def angle_detect_dnn(img, adjust=True):
    """
    文字方向检测
    """
    h, w = img.shape[:2]
    ROTATE = [0, 90, 180, 270]
    if adjust:
        thesh = 0.05
        xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
        img = img[ymin:ymax, xmin:xmax]  ##剪切图片边缘

    inputBlob = cv2.dnn.blobFromImage(img,
                                      scalefactor=1.0,
                                      size=(224, 224),
                                      swapRB=True,
                                      mean=[103.939, 116.779, 123.68], crop=False);
    angleNet.setInput(inputBlob)
    pred = angleNet.forward()
    index = np.argmax(pred, axis=1)[0]
    return ROTATE[index]


def angle_detect_tf(img, adjust=True):
    """
    文字方向检测
    """
    h, w = img.shape[:2]
    ROTATE = [0, 90, 180, 270]
    if adjust:
        thesh = 0.05
        xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
        img = img[ymin:ymax, xmin:xmax]  ##剪切图片边缘
    img = cv2.resize(img, (224, 224))
    img = img[..., ::-1].astype(np.float32)

    img[..., 0] -= 103.939
    img[..., 1] -= 116.779
    img[..., 2] -= 123.68
    img = np.array([img])

    out = sess.run(predictions, feed_dict={inputImg: img,
                                           keep_prob: 0
                                           })

    index = np.argmax(out, axis=1)[0]
    return ROTATE[index]


def angle_detect(img, adjust=True):
    """
    文字方向检测
    """
    if opencvFlag == 'keras':
        return angle_detect_tf(img, adjust=adjust)
    else:
        return angle_detect_dnn(img, adjust=adjust)
