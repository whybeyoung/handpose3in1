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
import time

from detector import *
from onnx_model import ONNXModel
import os


class ItemsDetector(Detector):
    def __init__(self, model_path, model_name):
        self.model_name = model_name
        self.model_path = model_path
        self.model = ONNXModel(self.model_path, None)


if __name__ == "__main__":
    model = ItemsDetector("../models/items/yolov5l.onnx", "cc")
    path_ = "./test_data/"
    half = False
    for f_ in os.listdir(path_):
        if not f_.endswith("jpg"):
            continue
        model.model.detect_items(os.path.join(path_, f_))
        time.sleep(1000)
