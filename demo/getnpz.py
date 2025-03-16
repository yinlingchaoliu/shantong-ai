import requests
import os

# 下载数据集
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
response = requests.get(url)
open("data/mnist/mnist.npz", "wb").write(response.content)

# 解压数据集
with open("data/mnist/mnist.npz", "rb") as f:
    import numpy as np

    data = np.load(f)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
