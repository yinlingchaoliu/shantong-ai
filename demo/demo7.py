import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from holoviews import output
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义参数
input_size = 28  # 图像尺寸 28 *28 * 1 三维数据
num_classes = 10  # 标签种类数
num_epochs = 3  # 训练次数
batch_size = 64  # 大小批次

# 训练集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

# 测试集
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# 构建batch数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # conv + rule + pooling
        self.conv1 = nn.Sequential(  # 输入大小(batch,1,28,28)
            nn.Conv2d(  # 卷积层
                in_channels=1,  # 输入通道数  灰度层
                out_channels=16,  # 输出通道数
                kernel_size=5,  # 卷积核大小 5*5 特征提取
                stride=1,  # 步长
                padding=2),  # 填充 padding = (kernel_size - 1) /2
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2)  # 池化层
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # 上一层输出
                out_channels=32,  # 32输出
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # (h-f+2p/s) +1 = 28 - 5 + 2*2 + 1 = 28
        # 14 -5 + 2*2 +1 =14
        # pool h/2 w/2

        # 输入 28*28*1 -> conv1 28*28*16 -> pool(长宽减半) 14*14*16
        # ->conv2 14*14*32 -> pool 7*7*32
        # 全连接层 图转化为向量->才能给全连接层
        self.out = nn.Linear(7 * 7 * 32, 10)  # 全连接层

        # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

 # 准确率评估
def accuracy(predictions, labels):  # 预测值和真实值
    pred = torch.max(predictions.data, 1)[1]  # 预测值
    rights = pred.eq(labels.data.view_as(pred)).sum()  # 正确个数和总数
    return rights, len(labels)


# 训练网络模型

net = CNN()
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(num_epochs):
    train_rights = []
    for batch_idx, (data, target) in enumerate(train_loader):  # 遍历训练集
        net.train() # 训练模式
        output = net(data) # 输出
        loss = criterion(output, target) # 损失
        optimizer.zero_grad() # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step() # 步数更新
        right = accuracy(output, target) # 预测值和真实值
        train_rights.append(right) # 预测值和真实值

        # 测试验证 每隔100次验证一次
        if batch_idx % 100 == 0:
            net.eval() # 验证模式
            test_rights = []
            for (data, target) in test_loader:
                output = net(data)
                right = accuracy(output, target)
                test_rights.append(right)

            train_r = (sum([tup[0] for tup in train_rights]),sum([tup[1] for tup in train_rights]))
            test_r = (sum([tup[0] for tup in test_rights]), sum([tup[1] for tup in test_rights]))

            # 进度取整数
            # 损失取后6位小数
            # 准确率取后2位小数
            print(f"当前epoch:{epoch} [{batch_idx * batch_size}/{len(train_loader.dataset)}] "
                  f"({round(100. * batch_idx/len(train_loader))}%)\t 损失: {loss.data:.6f}\t"
                  f"训练集准确率 {100. * train_r[0] / train_r[1]:.2f}%\t"
                  f"测试集正确率 {100. * test_r[0] / test_r[1]:.2f}%"  )