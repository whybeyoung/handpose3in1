# handpose3in1

## Debug


### 

#### 识别手模型转换
python hand_onnx_export.py --weights /workspace/handpose3in1/models/onnx/gpu/hand_m.pt


#### 识别手21点检测模型转换

python model2onnx.py --model_path /workspace/handpose3in1/models/handpose/resnet_50-size-256-loss-0.0642.pth --model resnet_50  --GPUS 0


#### yolov5l 预训练模型转换

python  export.py --weights yolov5l.pt --include onnx 

