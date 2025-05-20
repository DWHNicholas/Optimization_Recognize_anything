# -*- coding: utf-8 -*-
# @Time:2025/5/19 15:59
# @software:PyCharm.
import os.path as osp
import torch
from ram.models import ram_plus,ram
import onnx
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

device = "cuda"
image_size = 384
ckpt_file = "pretrained/ram_plus_swin_large_14m.pth"


def export_onnx(onnx_file):
    if not osp.exists(onnx_file):
        model = ram_plus(pretrained=ckpt_file, image_size=image_size, vit="swin_l")
        # model = ram(pretrained=ckpt_file, image_size=image_size, vit="swin_l")   自行切换处理ram还是ram plus
        model.eval()
        model = model.to(device)
        image = torch.randn(1, 3, image_size, image_size).to(device)
        dynamic_axes = {"targets": {0: "batch_size"}, "bs": {0: "batch_size"}}
        torch.onnx.export(
            model,
            image,
            onnx_file,
            verbose=True,
            opset_version=16,
            export_params=True,
            input_names=["img"],
            dynamic_axes=dynamic_axes,
            output_names=["targets", "bs"],
        )
        # Optional: Verify the ONNX model using onnx.checker
        onnx.checker.check_model(onnx.load(onnx_file))


    ## 量化模型部分
    # onnx_version = tuple(map(int, onnx.__version__.split(".")))
    # assert onnx_version >= (
    #     1,
    #     14,
    #     0,
    # ), f"The onnx version must be large equal than '1.14.0', but got {onnx_version}"
    # print(f"Quantizing model and writing to {model_output}...")
    # quantize_dynamic(
    #     model_input=onnx_file,
    #     model_output=model_output,
    #     per_channel=False,
    #     reduce_range=False,
    #     weight_type=QuantType.QUInt8,
    # )

onnx_file = "pretrained/ram_plus.onnx"
export_onnx(onnx_file)
