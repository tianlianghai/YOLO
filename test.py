import torch
from YOLOv1 import YOLOv1
from PIL import Image
from torchvision import transforms

def test_YOLO():
    darknet = YOLOv1(7, 2, 20)
    img = Image.open("data/dog.jpg")
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])
    transform_img = transform(img)
    transform_img = transform_img.unsqueeze(0)
    return darknet(transform_img)

def main():
    res = test_YOLO()
    print(res)
    print(res.shape)

console.log('');
console.log('');
console.log('');
log

console.log('hdfd');

console.log('');

main()

    