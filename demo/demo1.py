import torch


# 矩阵基础用法

# 创建矩阵 5*3
def empty():
    x = torch.empty(5, 3)
    print(x)


# 随机矩阵
def rand():
    x = torch.randn(5, 3)
    print(x)


# 全0矩阵
def zero():
    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)


# 传入数据 numpy
def tensor():
    x = torch.tensor([5.5, 3])
    print(x)


# 创建矩阵
def new_ones():
    x = torch.randn(5, 3)
    x = x.new_ones(5, 3, dtype=torch.double)
    x = torch.rand_like(x, dtype=torch.float)
    print(x)


# 创建矩阵大小
def size():
    x = torch.randn(5, 3)
    print(x.size())


# 矩阵加法
def add():
    x = torch.randn(5, 3)
    y = torch.randn(5, 3)
    print(x + y)
    # 或
    torch.add(x, y)


# 待验证
def index():
    x = torch.randn(5, 3)
    print(x[:, 1])  # 1行


# view操作改变矩阵
def view():
    x = torch.randn(4, 4)
    y = x.view(16)
    # -1 自动计算
    z = x.view(-1, 8)
    print(x.size(), y.size(), z.size())


# tensor 转 numpy
def toNumpy():
    a = torch.ones(5)
    b = a.numpy()
    print(b)


def toTensor():
    import numpy as np
    a = np.ones(5)
    b = torch.from_numpy(a)
    print(b)


empty()

rand()

zero()

tensor()

new_ones()

size()

add()

view()

toNumpy()

toTensor()
