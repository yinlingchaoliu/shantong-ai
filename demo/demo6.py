from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

# python + 翻墙
URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

import pickle
import gzip

# 读取文件 将路径转换为POSIX格式，确保跨平台兼容性。
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    data = pickle.load(f, encoding="latin-l")
    # 将数据拆解 成 训练集 、验证集 、测试集
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data
# 查看数据集的形状
print("训练集大小:", x_train.shape)
print("验证集大小:", x_valid.shape)
print("测试集大小:", x_test.shape)

import matplotlib.pyplot as plt

plt.imshow(x_train[0].reshape((28, 28)), cmap='gray')
plt.show()

import torch

# 数据转换成tensor
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
# n,c = x_train.shape
# x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train.shape)
print(y_train)
print(y_train.min(), y_train.max())

# 有bug 暂时忽略
import torch.nn.functional as F
import numpy as np

loss_func = F.cross_entropy

bs = 64
xb = x_train[0:bs]
yb = y_train[0:bs]
weights = torch.randn([784, 10], dtype=torch.float, requires_grad=True)
bs = 64
bias = torch.zeros(10, requires_grad=True)

print(xb)
print(yb)


def model(xb):
    return xb.mm(weights) + bias


# loss_func(model(xb),yb)


from torch import nn


class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        # 28*28 全链接层
        self.hidden1 = nn.Linear(784, 128).float()
        self.hidden2 = nn.Linear(128, 256).float()
        self.out = nn.Linear(256, 10).float()

    # 前向传播 (反向传播 torch框架自动处理)
    def forward(self, x):
        # 激活函数
        x = F.relu(self.hidden1(x)).float()
        x = F.relu(self.hidden2(x)).float()
        x = self.out(x).float()
        return x


net = Mnist_NN()
print(net)

# 打印网络参数
for name, param in net.named_parameters():
    print(name, param, param.size())

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

bs = 64

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

vaild_ds = TensorDataset(x_valid, y_valid)
vaild_dl = DataLoader(vaild_ds, batch_size=bs * 2)


# 方便batch取数据
def get_data(train_ds, vaild_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(vaild_ds, batch_size=bs * 2)
    )


import torch.nn.functional as F
import numpy as np

loss_func = F.cross_entropy


# model.train() # 训练模式 会使用Batch Normalization 和 Dropout
# model.eval() # 验证模式 不会使用

# 训练函数
def fit(steps, model, loss_func, opt, train_dl, vaild_dl):
    for step in range(steps):
        model.train()  # 训练模式
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()  # 验证模式
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in vaild_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(f"当前{step}:验证集损失{val_loss}")


from torch import optim


def get_model():
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=0.001)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


# 实际执行
bs = 64
loss_func = F.cross_entropy
train_ds = TensorDataset(x_train, y_train)
vaild_ds = TensorDataset(x_valid, y_valid)

train_dl, vaild_dl = get_data(train_ds, vaild_ds, bs)
model, opt = get_model()
fit(25, model, loss_func, opt, train_dl, vaild_dl)
