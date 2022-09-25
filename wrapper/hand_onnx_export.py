"""Exports a pytorch *.pt model to *.onnx format

Usage:
    import torch
    $ export PYTHONPATH="$PWD" && python models/onnx_export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse

import onnx

from models.common import *
import torch
from utils import torch_utils
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')

    parser.add_argument('--half', action='store_true', default=False, help='batch size')
    opt = parser.parse_args()
    print(opt)

    # Parameters
    f = opt.weights.replace('.pt', '.onnx')  # onnx filename
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size, (1, 3, 320, 192) iDetection

    # Load pytorch model
    google_utils.attempt_download(opt.weights)
    device = torch_utils.select_device("")
    img = img.to(device)
    model = torch.load(opt.weights, map_location=torch.device(device))['model']
    half = False
    if opt.half:
        if device == "cpu":
            raise Exception("not support half in cpu")
        half = True
        img, model = img.half(),model.half()

    model.eval()
    model.fuse()
    model.to(device)

    # Export to onnx
    model.model[-1].export = True  # set Detect() layer export=True
    model = model
    print(model.names)
    print("device:",device)

    _ = model(img)  # dry run
    torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
                      output_names=['output'])  # output_names=['classes', 'boxes']

    # Check onnx model
    model_onnx = onnx.load(f)  # load onnx model
    d = { "stride": int(max(model.stride)), "names": model.names}
    # Metadata
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)

    onnx.checker.check_model(model_onnx)  # check onnx model
    onnx.save(model_onnx, f)
    print(onnx.helper.printable_graph(model_onnx.graph))  # print a human readable representation of the graph
    print('Export complete. ONNX model saved to %s\nView with https://github.com/lutzroeder/netron' % f)
