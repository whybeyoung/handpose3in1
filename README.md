# handpose3in1

## 概要

将 yolov5识别手模型 和 handpose-x模型，标准yolov5-发布80类yolov5l模型用onnx推理合为1个代码推理

## demo

### demo1
https://user-images.githubusercontent.com/10629930/192127443-3116497b-e1f6-423d-8be9-8b379dc3ff68.mp4

### demo2
![img](demo.gif)

### 用法

请查看 [engine.py](wrapper/engine.py)内部test方法实现

### 模型转换部分代码

#### 识别手模型转换
python hand_onnx_export.py --weights /workspace/handpose3in1/models/onnx/gpu/hand_m.pt

#### 识别手21点检测模型转换

python model2onnx.py --model_path /workspace/handpose3in1/models/handpose/resnet_50-size-256-loss-0.0642.pth --model resnet_50  --GPUS 0

#### yolov5l 预训练模型转换

python  export.py --weights yolov5l.pt --include onnx 

#### 备注

部分前后处理代码,未做调整整合，仅供参考

#### 鸣谢

感谢 [yolov5](https://github.com/ultralytics/yolov5/releases)项目!
感谢 [handpose-x](https://github.com/EricLee2021-72324/handpose_x)项目!