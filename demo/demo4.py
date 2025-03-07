import torch
from torch import tensor

# tensor 常见格式

# 数值
def scalar():
    x = tensor(42.)
    print(x)
    print(x.dim())
    print(2*x)
    print(x.item())

#向量 深度学习中通常指特征 词向量特征 样本(身高, 体重, 年龄) 一个值的集合  1维度
def vector():
    v = tensor([1.5,-0.5,3.0])
    print(v)
    print(v.dim())
    print(v.size())


#矩阵  多个特征组合在一起   2维度
def matrix():
    m = tensor([[1, 2], [3, 4]])
    print(m)
    print(m.matmul(m))
    print(tensor([1,0]).matmul(m))
    print(m*m)


#高纬特征
def n_dimensional():
    x = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(x)


if __name__ == '__main__':
    # scalar()
    # vector()
    matrix()