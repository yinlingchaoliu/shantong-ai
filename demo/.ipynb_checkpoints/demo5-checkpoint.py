from pyexpat import features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings

import os

from sympy.physics.units import years

warnings.filterwarnings("ignore")
print(os.getcwd())
os.chdir(os.getcwd())
features = pd.read_csv('temps.csv')

#打印展示数据表
print(features.head())

print('数据维度:',features.shape)


# 处理数据 分别得到 年 月 日
import datetime
# 取出数据列
years = features['year']
months = features['month']
days = features['day']

dates = [str(int(year))+'-'+str(int(month))+'-'+str(int(day)) for year ,month, day in zip(years, months,days)]
print(dates[:5])
#数据格式转换
dates = [datetime.datetime.strptime(date,'%Y-%m-%d') for date in dates]
# 打印前5条数据
print(dates[:5])


#准备画图
plt.style.use('fivethirtyeight')

#设置布局
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
fig.autofmt_xdate(rotation=45)

#标签值
ax1.plot(dates,features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature');ax1.set_title('Max Temp')

ax2.plot(dates,features['temp_1'])
ax2.set_xlabel(''); ax2.set_ylabel('Temperature');ax2.set_title('Pre Temp')

ax3.plot(dates,features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature');ax3.set_title('Two Pre')


ax4.plot(dates,features['friend'])
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature');ax4.set_title('Friend')

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
features = features.drop('actual',axis=1)

# 名字单独保存一下，以备后患
feature_list = list(features.columns)
print(feature_list)

features = np.array(features)

print(features)
print(features.shape)


# 构建网络模型

# 预处理 - 标准化操作
from sklearn import preprocessing
input_features = preprocessing.StandardScaler().fit_transform(features)


x = torch.tensor(input_features,dtype=float)

