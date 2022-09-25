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
from utils.dataloaders import *

def increment_path(path, exist_ok=True, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


class Engine(object):
    def __init__(self):
        self.items_model = ItemsDetector("../models/onnx/cpu/yolov5l.onnx", "cc")
        self.hand_model = HandDetector("../models/onnx/cpu/hand_m.onnx", "c2c")
        self.h21_model = Hand21Detector("../models/onnx/cpu/resnet_50_size-256.onnx")

    def load_source(self, source, imgsz=[640, 640], stride=32, pt=True, vid_stride=1,save_dir="./test_data/output"):
        # Directories
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0]
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
        name = "exp"

        project = "test_data/output"

        save_dir = increment_path(Path(project) / name, exist_ok=True)

        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = 1
        vid_path, vid_writer = [None] * bs, [None] * bs
        i = 0
        for path, im, im0s, vid_cap, s in dataset:
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            self.detect(im0, p)
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0)



    def detect(self, srcimg,p="imge",vis=True):
        # 1. 检测手位置
        hand_loc, det1 = self.hand_model.model.detect_hand_(srcimg, False)

        # 1.5 检测手21点:
        # 1.5.1 取置信度最大的手部  bbox
        # 1.5.2 截图并预处理
        # 1.5.3 推理
        # 1.5.4 返回21点相对bbox位置，计算原图位置
        handpos_loc = self.h21_model.pre_detect(srcimg, hand_loc, False)

        # 2.检测所有物体(目前只要80类demo)
        items_loc, det2 = self.items_model.model.detect_items_(srcimg, False)

        # 统一画图
        if det1 is not None and len(det1):
            self.hand_model.model.draw(srcimg, det1, vis=False)
        if det2 is not None and len(det2):
            self.items_model.model.draw(srcimg, det2, vis=False)
        # ------------- 绘制关键点----------
        if handpos_loc:
            for k, v in handpos_loc.items():
                x = v['x']
                y = v['y']
                cv2.circle(srcimg, (int(x), int(y)), 3, (255, 50, 60), -1)
                cv2.circle(srcimg, (int(x), int(y)), 1, (255, 150, 180), -1)

            draw_bd_handpose(srcimg, handpos_loc, 0, 0)
        if vis: 
            cv2.imshow(str(p), srcimg)
            cv2.waitKey(1)  # 1 millisecond
        result = {"hand": hand_loc, "items": items_loc, "handpose": handpos_loc}
       # print(result)
        return result

    def test(self):
        path_ = "./test_data/"
        half = False
        for f_ in os.listdir(path_):
            if not f_.endswith(".png") and not f_.endswith(".jpg"):
                continue
            srcimg = cv2.imread(os.path.join(path_, f_))
            self.detect(srcimg,vis=False)

    def test_mp4(self, source):
        self.load_source(source)


if __name__ == '__main__':
    e = Engine()
    import time
    e.test_mp4("test_data/test-2.mp4")

    # while True:
    #     time.sleep(1)
    #
    #     #e.test()
