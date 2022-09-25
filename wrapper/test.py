#!/usr/bin/env python
# coding:utf-8
""" 
@author: nivic ybyang7
@license: Apache Licence 
@file: test.py
@time: 2022/09/22
@contact: ybyang7@iflytek.com
@site:  
@software: PyCharm 

# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛ 
"""

#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
from hand_onnx_detect import HandDetector
from handpose_detect import Hand21Detector
from items_detect import ItemsDetector
from utils.cv_utils import draw_bd_handpose
import os
import cv2


if __name__ == '__main__':
    items_model = ItemsDetector("../models/items/yolov5l.onnx", "cc")
    hand_model = HandDetector("../models/hand/hand_m.onnx", "c2c")
    h21_model = Hand21Detector("../models/handpose/resnet_50_size-256.onnx")
    path_ = "./test_data/"
    half = False
    for f_ in os.listdir(path_):
        if not f_.endswith(".png") and not f_.endswith(".jpg"):
            continue
        srcimg = cv2.imread(os.path.join(path_, f_))
        # 1. 检测手位置
        hand_loc, det1 = hand_model.model.detect_hand_(srcimg, False)

        print(hand_loc)
        # 1.5 检测手21点:
        # 1.5.1 取置信度最大的手部  bbox
        # 1.5.2 截图并预处理
        # 1.5.3 推理
        # 1.5.4 返回21点相对bbox位置，计算原图位置
        handpos_loc = h21_model.pre_detect(srcimg, hand_loc, False)

        # 2.检测所有物体(目前只要80类demo)
        items_loc, det2 = items_model.model.detect_items_(srcimg, False)

        # 统一画图
        hand_model.model.draw(srcimg, det1, vis=False)
        items_model.model.draw(srcimg, det2, vis=False)
        # ------------- 绘制关键点----------
        for k, v in handpos_loc.items():
            x = v['x']
            y = v['y']
            cv2.circle(srcimg, (int(x), int(y)), 3, (255, 50, 60), -1)
            cv2.circle(srcimg, (int(x), int(y)), 1, (255, 150, 180), -1)

        draw_bd_handpose(srcimg, handpos_loc, 0, 0)

        cv2.imshow('image', srcimg)
        cv2.waitKey(100000)
        result = {"hand": hand_loc, "items": items_loc, "handpose": handpos_loc}
        # print(result)
