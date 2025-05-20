# -*- coding: utf-8 -*-
# @Time:2025/5/19 18:09
# @software:PyCharm
import os

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from ram import get_transform

def convert_to_rgb(image):
    return image.convert("RGB")

def get_torchvision_transform(image_size):
    return transforms.Compose([
        convert_to_rgb,
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class RAM:
    def __init__(self, model_config=None) -> None:
        self.config = model_config
        model_path = self.config["model_path"]

        # 初始化 ONNX 推理会话
        self.sess_opts = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

        self.providers = ["CUDAExecutionProvider" if self.config["device"].lower() == "gpu"
                          else "CPUExecutionProvider"]

        self.ort_session = ort.InferenceSession(
            model_path,
            providers=self.providers,
            sess_options=self.sess_opts,
        )

        # 加载标签文件
        self.tag_list = self._load_tag_list(self.config["tag_list"])
        self.tag_list_chinese = self._load_tag_list(self.config["tag_list_chinese"])

        # 获取模型输入信息
        self.input_shape = self.ort_session.get_inputs()[0].shape[-2:]
        self.delete_tag_index = []

    def preprocess(self, image):
        transform = get_torchvision_transform(self.input_shape[0])
        blob = transform(Image.open(image))
        blob = blob.unsqueeze(0).numpy()
        return blob

    def predict(self, image):
        """完整推理流程"""
        blob = self.preprocess(image)
        outputs = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: blob})
        return self._postprocess(outputs)

    def _postprocess(self, outputs):
        """后处理输出结果"""
        tags, bs = outputs
        tags[:, self.delete_tag_index] = 0

        results = []
        for b in range(bs[0]):
            indices = np.argwhere(tags[b] == 1).squeeze(1)
            results.extend(self.tag_list_chinese[indices].tolist())
        return results

    @staticmethod
    def _load_tag_list(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return np.array(f.read().splitlines())



img_path = "images/demo/demo1.jpg"
tag_list = "ram/data/ram_tag_list.txt"
tag_list_chinese = "ram/data/ram_tag_list_chinese.txt"
onnx_model = "pretrained/img2tags_plus.onnx"
configs = {
    "model_path":onnx_model,
    "device":"cpu",
    "delete_tag_index": [],
    "tag_list": tag_list,
    "tag_list_chinese": tag_list_chinese,
}
model = RAM(configs)

res = model.predict(img_path)
print(res)