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

from onnx_model import ONNXModel


class HandDetector(Detector):
    def __init__(self, model_path, model_name):
        self.model = ONNXModel(model_path, classes=["hand"], gpu_cfg=True)

    def detect(self, img0: numpy.ndarray):
        output_dict_ = self.model.infer(img0)

        return output_dict_


if __name__ == "__main__":
    model = HandDetector("../models/hand/hand_m.onnx", "c2c")
    path_ = "./test_data/"
    half = False
    for f_ in os.listdir(path_):
        model.model.detect_hand(os.path.join(path_, f_))
