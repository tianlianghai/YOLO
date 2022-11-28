import torch
from YOLOv1 import YOLOv1
from YoloLoss import YoloLoss
from PIL import Image
from torchvision import transforms
from Dataset import VOCDataset
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

def test_dataset():
    img_dir = "data/images"
    label_dir = "data/labels"
    # csv_dir = "data/train.csv"
    csv_dir = "data/8examples.csv"
    dataset = VOCDataset(img_dir, label_dir, csv_dir)
    return dataset

def main():
    res = test_dataset()
    res[0][0].show()
    print(res[1].shape)

main()