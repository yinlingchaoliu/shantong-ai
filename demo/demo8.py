import torchvision.models as models
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from altair import data_transformers
from torch import nn
import torch.optim as optim
from torch.cuda import device
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
from torch.utils.data import DataLoader

# 读取数据
data_dir = './data/flower_data/'
train_dir = data_dir + 'train/'
vaild_dir = data_dir + 'valid/'

# 数据增强 (1.数据量变多(一个变多个) 2.高效利用数据)
# 制作数据源
data_transformers = {
    # 数据增强
    'train': transforms.Compose([
        # transforms.Resize(256),  # 一般要做resize操作
        transforms.RandomRotation(45),  # 随机旋转 -45~45
        transforms.CenterCrop(224),  # 从中心剪裁
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(p=0.025),  # 随机灰度
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化 均值和标准差
    ]),
    # 验证集 不需要做数据增强
    'valid': transforms.Compose([
        transforms.Resize(256),  # 缩放
        transforms.CenterCrop(224),  # 从中心剪裁
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化 是一套数据 均值和标准差
    ])
}

# batch数据制作 输入大小影响训练速度
batch_size = 8
# 224 * 224 *3(rgb)
# 迭代次数
# batch_size

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transformers[x]) for x in ['train', 'valid']}

# 展开是这句话
# image_datasets = {
#     'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transformers['train']),
#     'valid': datasets.ImageFolder(os.path.join(data_dir, 'valid'), data_transformers['valid'])
# }

# 数据加载器
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
               ['train', 'valid']}

# 展开
# dataloaders={
#     'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
#     'valid': DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=True)
# }

dataset_size = {x: len(image_datasets[x]) for x in ['train', 'valid']}

class_names = image_datasets['train'].classes

print(image_datasets)

# 读取标签对应的名字

with open('data/flower_data/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    print(cat_to_name)


# 展示数据
def im_convert(tensor):
    """ 展示数据 """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))  # 反归一化
    image = image.clip(0, 1)
    return image


fig = plt.figure(figsize=(20, 12))
columns = 4
rows = 2
dataiter = iter(dataloaders['valid'])
inputs, classes = dataiter.__next__()

for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
    plt.imshow(im_convert(inputs[idx]))

plt.show()

# https://pytorch.org/vision/stable/models.html#classification
# resnet paper https://arxiv.org/abs/1512.03385
# pdf 论文 https://arxiv.org/pdf/1512.03385

# 加载训练好的模型 进行定制  主流使用resnet 处理
model_name = 'resnet'  # vgg resnet
# 是否可以训练
feature_extract = True

# 是否可以gpu训练
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('GPU is not available')
else:
    print('GPU is available')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 迁移学习使用 : 指定层不做训练更新
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


model_ft = models.resnet152()
# 打印模型特征
print(model_ft)


# 加载好训练模型 (迁移学习)
#  (fc): Linear(in_features=2048, out_features=1000, bias=True)
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        # 梯度更新 false 相当于 这一层失效 冻住
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features  # 获取输出层值 2048
        # 修改输出 从 1000 到 102
        # fc 这一层重新定义 即迁移学习
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102),  # 改输出层为 102
                                    nn.LogSoftmax(dim=1))  # 损失打印
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
print(model_ft)

model_ft = model_ft.to(device)

filename = './model/reset152.pth'

params_to_update = model_ft.parameters()

print('Params to learn')
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print('\t', name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print('\t', name)

# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) # 学习率 每7个epoch 衰减原来1/10
criterion = nn.NLLLoss()

# 训练模块
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False,filename=filename):
    since = time.time()
    best_acc = 0
