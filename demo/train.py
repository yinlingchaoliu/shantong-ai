from pathlib import Path
import torch
import torch.nn.functional as F

from torch import nn
import pickle
import gzip

import numpy as np

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim


# 读取训练集
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"

# 读取文件 将路径转换为POSIX格式，确保跨平台兼容性。
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    data = pickle.load(f, encoding="latin-l")
    # 将数据拆解 成 训练集 、验证集 、测试集
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data
# 查看数据集的形状
print("训练集大小:", x_train.shape)
print("验证集大小:", x_valid.shape)
print("测试集大小:", x_test.shape)

# 展示图片
import matplotlib.pyplot as plt
plt.imshow(x_train[0].reshape((28, 28)), cmap='gray')
plt.show()

class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        # 28*28 全链接层
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    # 前向传播 (反向传播 torch框架自动处理)
    def forward(self, x):
        # 激活函数
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x

net = Mnist_NN()
print(net)

# 打印网络参数
for name, param in net.named_parameters():
    print(name, param, param.size())


# 方便batch取数据
def get_data(train_ds, vaild_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(vaild_ds, batch_size=bs * 2)
    )


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

# 获取模型和优化器
def get_model():
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=0.001)

# 损失函数和反向传播
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)



# 实际执行
weights = torch.randn([784, 10], dtype=torch.float, requires_grad=True)
bias = torch.zeros(10, requires_grad=True)

bs = 64
loss_func = F.cross_entropy

# 数据转换成tensor
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

train_ds = TensorDataset(x_train, y_train)
vaild_ds = TensorDataset(x_valid, y_valid)

train_dl, vaild_dl = get_data(train_ds, vaild_ds, bs)
model,opt = get_model()
fit(25, model, loss_func, opt, train_dl, vaild_dl)
