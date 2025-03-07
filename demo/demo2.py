import torch


# 框架反向传播自动做好
# 求导

# 构建3行4列矩阵
def matrix():
    x = torch.randn(3, 4, requires_grad=True)
    b = torch.randn(3, 4, requires_grad=True)
    print(x)
    print(b)
    t = x + b
    print(t)
    y = t.sum()
    print(y)
    # 反向传播
    y.backward()
    print(b.grad)
    print(x.requires_grad, b.requires_grad, t.requires_grad)


def matrix1():
    x = torch.randn(1)
    b = torch.randn(1, requires_grad=True)
    w = torch.randn(1, requires_grad=True)
    y = x * w
    z = y + b
    print(x.requires_grad, b.requires_grad, w.requires_grad, y.requires_grad)
    print(x.is_leaf, b.is_leaf, w.is_leaf, y.is_leaf)
    z.backward(retain_grad=True)
    print(w.grad)




matrix()

# w * x = y
# y + b = z
