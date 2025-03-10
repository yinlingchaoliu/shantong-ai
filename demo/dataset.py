from pathlib import Path
import requests

# 需求背景与原代码问题
# 你之前的代码尝试从 OpenML 下载 MNIST 数据集并将其保存为 mnist.pkl.gz 格式。不过原代码直接将整个数据集保存，没有区分训练集、验证集和测试集。现在需求是要支持将数据集划分为 (x_train, y_train), (x_valid, y_valid), (x_test, y_test) 并进行保存。
#
# 实现思路
# 下载数据集：使用 fetch_openml 从 OpenML 下载 MNIST 数据集。
# 划分数据集：将数据集划分为训练集、验证集和测试集。一般来说，MNIST 数据集包含 70000 个样本，常见的划分是 60000 个训练样本和 10000 个测试样本，这里我们可以从训练样本中再划分一部分作为验证集。
# 保存数据集：将划分好的数据集保存为 mnist.pkl.gz 格式。
#获取MNIST数据集
import pickle
import gzip
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 下载数据集
mnist = fetch_openml('mnist_784', version=1)
x, y = mnist['data'], mnist['target']

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=10000 / 70000, random_state=42)

# 从训练集中划分验证集
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# 封装数据
data = ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))

# 保存为 mnist.pkl.gz  格式
with gzip.open('data/mnist/mnist.pkl.gz', 'wb') as f:
    pickle.dump(data, f)


# 验证数据集
import pickle
import gzip

with gzip.open('data/mnist/mnist.pkl.gz', 'rb') as f:
    data = pickle.load(f,  encoding='latin-1')

 # 分割数据集为训练集、验证集和测试集
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data

# 查看数据集的形状
print("训练集大小:", x_train.shape)
print("验证集大小:", x_valid.shape)
print("测试集大小:", x_test.shape)


# 下载并加载 MNIST 数据集
from sklearn.datasets import fetch_openml
import pickle
import gzip

# 获取MNIST数据集  未划分
# mnist = fetch_openml('mnist_784', version=1)
# x, y = mnist['data'], mnist['target']
#
# # 保存为mnist.pkl.gz 格式
# with gzip.open('mnist.pkl.gz', 'wb') as f:
#     pickle.dump((x, y), f)