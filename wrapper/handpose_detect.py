#!/usr/bin/env python
# coding:utf-8
""" 
@author: nivic ybyang7
@license: Apache Licence 
@file: hand_detect.py
@time: 2022/09/21
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


# -*-coding:utf-8-*-
# function: onnx Inference
import os, sys
import time

import onnxruntime
import cv2
import numpy as np
from utils.cv_utils import draw_bd_handpose
from detector import *

from utils.torch_utils import select_device



class Hand21Detector(Detector):
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        providers = ['CUDAExecutionProvider']
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        print("{}: {}".format(os.path.basename(onnx_path),self.onnx_session.get_providers()))
        providers = self.onnx_session.get_providers()
        self.fp16 = True if 'CUDAExecutionProvider' in providers else False
        self.device = select_device("")

        meta = self.onnx_session.get_modelmeta().custom_metadata_map

        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output

    def draw(self, img0, pts_hand, output):
        img_width = img0.shape[1]
        img_height = img0.shape[0]
        draw_bd_handpose(img0, pts_hand, 0, 0)  # 绘制关键点连线

        # ------------- 绘制关键点
        for i in range(int(output.shape[0] / 2)):
            x = (output[i * 2 + 0] * float(img_width))
            y = (output[i * 2 + 1] * float(img_height))

            cv2.circle(img0, (int(x), int(y)), 3, (255, 50, 60), -1)
            cv2.circle(img0, (int(x), int(y)), 1, (255, 150, 180), -1)

        cv2.namedWindow('image', 0)
        cv2.imshow('image', img0)
        if cv2.waitKey(60000) == 27:
            return

    def detect_hand_21(self, img0, vis=False):
        img_size = 256

        img = cv2.resize(img0, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

        img_ndarray = img.transpose((2, 0, 1))
        img_ndarray = (img_ndarray - 128.) / 256.
        img_ndarray = np.expand_dims(img_ndarray, 0)
        img_ndarray = torch.from_numpy(img_ndarray).to(self.device)

        img_ndarray = img_ndarray.half() if False else img_ndarray.float()  # uint8 to fp16/32
        im = img_ndarray.cpu().numpy()
        output = self.forward(im)[0][0]
        output = np.array(output)
        return output

    def pre_detect(self, img0, hand_loc, vis=True):
        if not hand_loc:
            return
        hnd_loc = hand_loc[0]
        # *xyxy, conf, cls = hnd_loc
        x1, y1, x2, y2 = hnd_loc['bbox']
        # 增加保存对应图片ybyang7
        x_min, y_min, x_max, y_max = float(x1), float(y1), float(x2), float(y2)
        w_ = max(abs(x_max - x_min), abs(y_max - y_min))

        w_ = w_ * 1.1

        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2

        x1, y1, x2, y2 = int(x_mid - w_ / 2), int(y_mid - w_ / 2), int(x_mid + w_ / 2), int(
            y_mid + w_ / 2)

        x1 = np.clip(x1, 0, img0.shape[1] - 1)
        x2 = np.clip(x2, 0, img0.shape[1] - 1)

        y1 = np.clip(y1, 0, img0.shape[0] - 1)
        y2 = np.clip(y2, 0, img0.shape[0] - 1)
        hand_img = img0[y1:y2, x1:x2]
        output = self.detect_hand_21(hand_img, vis=False)
        pts_hand = {}  # 构建关键点连线可视化结构\
        img_width = hand_img.shape[1]
        img_height = hand_img.shape[0]
        # ------------- 记录关键点----------
        for i in range(int(output.shape[0] / 2)):
            x = (output[i * 2 + 0] * float(img_width)) + x1
            y = (output[i * 2 + 1] * float(img_height)) + y1
            pts_hand[str(i)] = {}
            pts_hand[str(i)] = {
                "x": x,
                "y": y,
            }

        if vis:
            # ------------- 绘制关键点----------
            for i in range(int(output.shape[0] / 2)):
                x = (output[i * 2 + 0] * float(img_width)) + x1
                y = (output[i * 2 + 1] * float(img_height)) + y1
                cv2.circle(img0, (int(x), int(y)), 3, (255, 50, 60), -1)
                cv2.circle(img0, (int(x), int(y)), 1, (255, 150, 180), -1)

            draw_bd_handpose(img0, pts_hand, 0, 0)

        return pts_hand


if __name__ == "__main__":
    img_size = 256
    model = Hand21Detector("../models/handpose/resnet_50_size-256.onnx")
    path_ = "./test_data/handpose/"
    for f_ in os.listdir(path_):
        if not f_.endswith(".png") and not f_.endswith("jpg"):
            continue

        img0 = cv2.imread(path_ + f_)

        pts_hand = model.detect_hand_21(img0, True)

        time.sleep(10000)
