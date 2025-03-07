import numpy as np
import torch
import torch.nn as nn


# 线性回归模型  数学公式 代码映射

def linear_regression():
    x_values = [i for i in range(10)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)
    print(x_train.shape)

    y_values = [2 * i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)
    print(y_train.shape)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(LinearRegressionModel, self).__init__()
        # 全链接层
        self.linear = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


def main():
    input_dim = 1
    out_dim = 1
    model = LinearRegressionModel(input_dim, out_dim)
    print(model)

    # todo 使用GPU进行训练  只需要把数据和模型传入到cuda里面即可
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #模型放入设备
    model.to(device)

    # 制定好参数和损失函数
    epochs = 1000  # 训练次数
    learning_rate = 0.01 # 学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # 优化器
    criterion = nn.MSELoss() # 损失函数

    #参数
    x_values = [i for i in range(10)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)
    print(x_train.shape)

    y_values = [2 * i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)
    print(y_train.shape)


    #训练模型
    for epoch in range(epochs):
        epoch += 1
        #转化为tensor  todo 传入 to(device)
        inputs = torch.from_numpy(x_train).to(device)
        labels = torch.from_numpy(y_train).to(device)

        # 梯度要清零 每一次迭代 (反向传播梯度会累加)
        optimizer.zero_grad()

        #前向传播
        outputs = model(inputs)

        #计算损失
        loss = criterion(outputs, labels)

        #反向传播
        loss.backward()

        #步数 更新权重参数
        optimizer.step()
        if epoch % 50 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))

        #测试模型预测结果
        predicted = model(torch.from_numpy(x_train).requires_grad_(True)).data.numpy()
        print(predicted)

        #模型保存和读取
        torch.save(model.state_dict(), 'model.pkl')
        model.load_state_dict(torch.load('model.pkl'))


if __name__ == '__main__':
    linear_regression()
    main()
