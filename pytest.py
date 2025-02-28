import sys
import tensorflow as tf
import torch
import pandas as pd
import sklearn
import cv2

# 输出 Python 版本信息
print(f"Python 版本: {sys.version}")

# 输出 TensorFlow 版本信息
try:
    print(f"TensorFlow 版本: {tf.__version__}")
except AttributeError:
    print("未安装 TensorFlow 库")

# 输出 PyTorch 版本信息
try:
    print(f"PyTorch 版本: {torch.__version__}")
except AttributeError:
    print("未安装 PyTorch 库")

# 输出 Pandas 版本信息
try:
    print(f"Pandas 版本: {pd.__version__}")
except AttributeError:
    print("未安装 Pandas 库")

# 输出 Scikit - learn 版本信息
try:
    print(f"Scikit - learn 版本: {sklearn.__version__}")
except AttributeError:
    print("未安装 Scikit - learn 库")

# 输出 OpenCV 版本信息
try:
    print(f"OpenCV 版本: {cv2.__version__}")
except AttributeError:
    print("未安装 OpenCV 库")