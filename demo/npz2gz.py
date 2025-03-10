import numpy as np
import pickle
import gzip

from sklearn.model_selection import train_test_split

# 加载 MNIST .npz 文件
mnist_npz = np.load('mnist.npz')

# 提取训练集、验证集和测试集的数据和标签
x_train = mnist_npz['x_train']
y_train = mnist_npz['y_train']

# x_valid = mnist_npz['x_valid']
# y_valid = mnist_npz['y_valid']

#测试集
x_test = mnist_npz['x_test']
y_test = mnist_npz['y_test']


#数据切割
# 从训练集中划分验证集 (x_train 取前面数据 , x_valid 取后面数据 不要随机)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1,shuffle=False)

# 关闭文件
mnist_npz.close()

# 定义数据结构
data = (
    (x_train, y_train),
    (x_valid, y_valid),
    (x_test, y_test)
)

# 保存为 .pkl.gz  文件
with gzip.open('mnist.pkl.gz', 'wb') as f:
    pickle.dump(data, f)