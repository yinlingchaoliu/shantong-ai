from pyexpat import features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings

import os

from openpyxl.styles.builtins import output
from skimage.feature import learn_gmm
from sympy.physics.units import years

warnings.filterwarnings("ignore")
print(os.getcwd())
os.chdir(os.getcwd())
features = pd.read_csv('temps.csv')

# 打印展示数据表
print(features.head())

print('数据维度:', features.shape)

# 处理数据 分别得到 年 月 日
import datetime

# 取出数据列
years = features['year']
months = features['month']
days = features['day']

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
print(dates[:5])
# 数据格式转换
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# 打印前5条数据
print(dates[:5])

# 准备画图
plt.style.use('fivethirtyeight')

# 设置布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.autofmt_xdate(rotation=45)

# 标签值
ax1.plot(dates, features['actual'])
ax1.set_xlabel('');
ax1.set_ylabel('Temperature');
ax1.set_title('Max Temp')

ax2.plot(dates, features['temp_1'])
ax2.set_xlabel('')
ax2.set_ylabel('Temperature')
ax2.set_title('Pre Temp')

ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date');
ax3.set_ylabel('Temperature');
ax3.set_title('Two Pre')

ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date');
ax4.set_ylabel('Temperature');
ax4.set_title('Friend')

plt.tight_layout(pad=2)

import matplotlib.pyplot as plt

plt.show()

# 独热编码
features = pd.get_dummies(features)
features.head(5)

print(features.head(5))

# 标签
labels = np.array(features['actual'])
# 在特征中去掉标签 去掉 actual 标签
features = features.drop('actual', axis=1)

# 名字单独保存一下，以备后患
feature_list = list(features.columns)
print(feature_list)

features = np.array(features)

print(features)
print(features.shape)
# [5,13] 5条数据 13个维度

# 构建神经网络模型

# 预处理 - 标准化操作
from sklearn import preprocessing

input_features = preprocessing.StandardScaler().fit_transform(features)

# np -> tensor
x = torch.tensor(input_features, dtype=float)
y = torch.tensor(labels, dtype=float)

# 数据输出[5,13]  13 代表 13个特征


#                  b1 128  b2 1
#                  w1 128  w2 [128,1]
# 数据输出[5,13] * [13,128] * [128,1]
# 隐藏特征 表示的神经元 128  13个输入特征转换成 128个隐藏特征
# 偏置参数(微调) b1 和隐藏特征数量一致
# w2 回归任务 1个参数
# b2 回归任务 1个参数 梯度计算

# 权重参数初始化 随机标准正态分布
weights = torch.randn((13, 128), dtype=float, requires_grad=True)
biases = torch.randn(128, dtype=float, requires_grad=True)
weights2 = torch.randn((128, 1), dtype=float, requires_grad=True)
biases2 = torch.randn(1, dtype=float, requires_grad=True)

learn_rate = 0.001
losses = []

for i in range(1000):  # 迭代 1000次
    # 计算隐层 x * w1 + b1
    hidden = x.mm(weights) + biases
    # 加入激活函数(权重参数要加上激活函数)
    hidden = torch.relu(hidden)
    # 预测结果  hidden * w2 + b2
    predictions = hidden.mm(weights2) + biases2

    # 计算损失 预测 - 真实 均方误差
    loss = torch.mean((predictions - y) ** 2)
    losses.append(loss.data.numpy())
    # 打印损失率
    if i % 100 == 0:
        print('loss:', loss)

    # 反向传播计算
    loss.backward()

    # 更新参数 更新梯度 - 反方向
    weights.data.add(-learn_rate * weights.grad.data)
    biases.data.add(-learn_rate * biases.grad.data)
    weights2.data.add(-learn_rate * weights2.grad.data)
    biases2.data.add(-learn_rate * biases2.grad.data)

    # 每次迭代都清空
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()

# 步骤
# 1.数据预处理
# 2.数据标准化
# 3.转化tensor
# 4.权重参数设计
# 5.前向传播 hidden= x.nn(w1) +b1
#          predictions = hidden.mm(w2) + b2
# 6.计算损失 loss
# 7.反向传播
# 7.计算梯度 更新
# 梯度置0
# 从前到后，再从后到前


# 更简单的构建网络模型

input_size = input_features.shape[1]  # 样本数量
hidden_size = 128  # 128 隐藏神经元
output_size = 1  # 输出 1 个结果
batch_size = 16  #
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size)
)

# 函数损失
cost = torch.nn.MSELoss(reduction='mean')
# 优化器  Adam 动态调节优化率
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

# 训练网络
losses = []
for i in range(1000):
    batch_loss = []
    # MINI-Batch 方法来进行训练
    for start in range(0, len(input_features), batch_size):
        # 3目运算符
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
        predictions = my_nn(xx)
        loss = cost(predictions, yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        batch_loss.append(loss.data.numpy())

        # 打印损失率
        if i % 100 == 0:
            print('loss:', loss)
            print(i, np.mean(batch_loss))

# 训练集合   测试集合切分

# 预训练结果
x = torch.tensor(input_features, dtype=torch.float)
predit = my_nn(x).data.numpy()

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
print(dates[:5])
# 数据格式转换
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

true_data = pd.DataFrame(data={'date': dates, 'actual': labels})

months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
              zip(years, months, days)]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predit.reshape(-1)})

plt.figure(figsize=(10,  6))

# 真实值
plt.plot(true_data['date'], true_data['actual'])
# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'])
plt.xticks()
plt.legend()

# 过拟合

plt.xlabel('Date')
plt.ylabel('Max F')
plt.title('Actual and Predicted Value')
plt.show()
