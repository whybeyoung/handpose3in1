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

import random

import numpy

from detector import *
from utils.datasets import *
from utils.utils import *


class HandDetector(Detector):
    def __init__(self, model_path, model_name):
        self.model_name = model_name
        self.model_path = model_path
        self.half = False  # half precision FP16 inference
        self.img_size = 640
        self.augment = False
        self.conf_thres = 0.31
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False  # 'class-agnostic NMS'

        # 初始化 模型推理硬件
        device = torch_utils.select_device("")
        self.device = device
        # 模型加载初始化
        self.model = torch.load(self.model_path, map_location=device)['model']
        # 模型设置为推理模式
        self.model.to(device).eval()
        # 模型设置为推理模式
        # Cpu 半精度 flost16 推理设置 ：Half precision
        half = self.half and device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            self.model.half()

        # Set Dataloader
        vid_path, vid_writer = None, None
        # Get names and colors
        self.names = self.model.names if hasattr(self.model, 'names') else self.model.modules.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.init_img()

    def init_img(self):
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img.float()) if self.device.type != 'cpu' else None  # run once

    def detect(self, img0: numpy.ndarray):
        self.img = letterbox(img0, new_shape=self.img_size)[0]
        self.img = self.img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        self.img = np.ascontiguousarray(self.img)

        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.half() if self.half else self.img.float()  # uint8 to fp16/32
        self.img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if self.img.ndimension() == 3:
            self.img = self.img.unsqueeze(0)
        # 模型推理
        t1 = torch_utils.time_synchronized()
        pred = self.model(self.img, augment=self.augment)[0]
        t2 = torch_utils.time_synchronized()
        if self.half:
            pred = pred.float()

        # NMS 操作
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                   fast=True, classes=self.classes, agnostic=self.agnostic_nms)
        # 输出检测结果
        output_dict_ = []
        for det in pred:  # detections per image

            s, im0 = '', img0
            # Write results

            s += '%gx%g ' % self.img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # 推理的图像分辨率转为原图分辨率：Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(self.img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                for i, (*xyxy, conf, cls) in enumerate(det):
                    x1, y1, x2, y2 = xyxy
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    output_dict_.append(
                        {"bbox": (float(x1), float(y1), float(x2), float(y2)),
                         "label": label,
                         "cls": self.names[int(cls)],
                         "confidence": float(conf)})
                    # 增加保存对应图片ybyang7
                    # x_min, y_min, x_max, y_max = float(x1), float(y1), float(x2), float(y2)
                    # w_ = max(abs(x_max - x_min), abs(y_max - y_min))
                    #
                    # w_ = w_ * 1.1
                    #
                    # x_mid = (x_max + x_min) / 2
                    # y_mid = (y_max + y_min) / 2
                    #
                    # x1, y1, x2, y2 = int(x_mid - w_ / 2), int(y_mid - w_ / 2), int(x_mid + w_ / 2), int(
                    #     y_mid + w_ / 2)
                    #
                    # x1 = np.clip(x1, 0, img0.shape[1] - 1)
                    # x2 = np.clip(x2, 0, img0.shape[1] - 1)
                    #
                    # y1 = np.clip(y1, 0, img0.shape[0] - 1)
                    # y2 = np.clip(y2, 0, img0.shape[0] - 1)
                    # print(x1, x2, y1, y2, str(i))
                    # cv2.imwrite("./inference/1-{}.jpg".format(str(i)), img0[y1:y2, x1:x2])
                    # plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
                print("Hand Detect output_dict_ : ", output_dict_)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))
        return output_dict_
