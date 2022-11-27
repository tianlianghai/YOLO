import torch
from YOLOv1 import YOLOv1
from YoloLoss import YoloLoss
from PIL import Image
from torchvision import transforms

darknet = YOLOv1()
img = Image.open("data/dog.jpg")
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
])
transform_img = transform(img)
transform_img = transform_img.unsqueeze(0)


def test_YOLO():
    
    return darknet(transform_img)

def test_yololoss():
    yololoss = YoloLoss()
    out = darknet(transform_img)
    target = torch.rand(1, 7, 7, 25)
    loss = yololoss(out, target)
    return loss


def main():
    res = test_yololoss()
    print(res)
    print(res.shape)

main()