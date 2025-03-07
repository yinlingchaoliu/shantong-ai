import torch
import torchvision.models  as models
import os

from streamlit.elements.image import image_to_url
from torchvision.models.segmentation  import deeplabv3_resnet50
import urllib
from PIL import Image
from torchvision import transforms

# https://pytorch.org/hub/
# https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_vision_deeplabv3_resnet101.ipynb

# hub仓库
# google search 看demo效果

#调用别人训练好的模型和模块
def loadModel():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    return model

def downloadImage(imageUrl,fileName):
    url, filename = (imageUrl, fileName)
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

#保存下载模型
def downloadModel():
    model = models.resnet50(pretrained=True)
    torch.save(model.state_dict(), 'resnet50.pth')

# 本地缓存加载再验证
def loadLocalModel():
    # 指定缓存路径（假设模型文件名为 resnet50.pth ）
    # 加载模型权重
    model_path = '~/.cache/torch/hub/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth'
    model_path = os.path.expanduser(model_path)
    model = deeplabv3_resnet50(pretrained=False)
    model.load_state_dict(torch.load(model_path, strict=False))
    # 设置为评估模式
    model.eval()
    print("模型加载成功！")

def listModels():
    expanded_path = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    print(os.listdir(expanded_path))

def transform(model,filename):
    # sample execution (requires torchvision)
    from PIL import Image
    from torchvision import transforms
    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions,input_image

def show(output_predictions,input_image):
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)

    import matplotlib.pyplot as plt
    plt.imshow(r)
    plt.show()

if __name__ == '__main__':
    #加载模型
    imageUrl = "https://github.com/pytorch/hub/raw/master/images/deeplab1.png"
    fileName = "deeplab1.png"

    model = loadModel()
    print(model)
    downloadImage(imageUrl, fileName)
    output_predictions, input_image = transform(model,fileName)
    show(output_predictions,input_image)