# Recognize_anything的优化

Recognize——anything的原始模型地址为：https://github.com/xinyu1205/recognize-anything.git

- 出现的问题：由于原始的模型很大，plus模型2.8G，ram模型5.63G，模型还位于huggingface上面，每次使用非常麻烦，并且安装包的过程由于墙的原因非常容易出现问题，使用过程非常不便捷，因此想要重新写一个onnx的推理版本，方便使用
- 优化思路：看了代码发现forward部分主要用于训练，直接拿来用于export的话，会存在一些问题，因此需要重新写一个forward部分，不过我没有写量化，由于暂时不需要，但是代码里面仍然给出了量化的部分，可以自行决定是否需要量化。

## 使用方式：
1. 下载[原始仓库](https://github.com/xinyu1205/recognize-anything.git) 和本仓库的代码，并将本仓库代码直接覆盖原始代码即可
2. 安装包，使用原始仓库的安装包即可：pip install -r requirements.txt
3. 安装完成后，需要下载原始模型到pretrained文件夹下,地址可以在原仓库中进行下载，自行决定要使用RAM 还是RAM_PLus,下载对应模型
4. 修改export.py文件，将模型路径修改为自己的模型路径，并切换RAM和RAM_PLus模型
5. 执行 python export_onnx.py即可
6. 执行完成后可以找到onnx模型，并使用onnxruntime进行推理，推理代码可以参考readonnx.py


## 使用过程中遇到的问题
1. 想要在执行onnx的时候将模型的处理转化为opencv的处理方式，但是这种预处理方案发现和基于tranformer的流式处理方案有一定的差别，结果有一点点的不同，为了验证猜想，用以下代码进行验证：
``` python 
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
def convert_to_rgb(image):
    return image.convert("RGB")
def get_torchvision_transform(image_size=384):
    return transforms.Compose([
        convert_to_rgb,
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_opencv(image_path, image_size=384):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")

    # 转换为 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 调整图像大小
    image = cv2.resize(image, (image_size, image_size))

    # 转换为浮点数并归一化到 [0, 1]
    image = image.astype(np.float32) / 255.0

    # 标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # 转换为 CHW 格式（通道优先）
    image = image.transpose(2, 0, 1)

    # 添加批次维度
    image = np.expand_dims(image, axis=0).astype(np.float32)

    return image


# 加载图像
image_path = "images/demo/demo1.jpg"
pil_image = Image.open(image_path)


torchvision_transform = get_torchvision_transform(384)
torchvision_result = torchvision_transform(pil_image).unsqueeze(0).numpy()

opencv_result = preprocess_opencv(image_path, 384)
print("Torchvision result shape:", torchvision_result.shape)
print("OpenCV result shape:", opencv_result.shape)
print("Difference:", np.abs(torchvision_result - opencv_result).max())
```
结果显示：
![yanzheng.jpg](images%2Fyanzheng.jpg)
因此为了避免这个问题，直接使用torchvision的transform进行预处理，但是如果你不介意的情况下，用opencv处理也是没有问题的，只是标签少一点而已。