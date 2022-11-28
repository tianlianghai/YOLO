import torch
from torch import optim
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from YOLOv1 import YOLOv1
from YoloLoss import YoloLoss
from Dataset import VOCDataset
from utils import(
    intersection_over_union,
    none_max_suppression,
    mean_average_precision,  
    load_checkpoint,
    save_checkpoint,
    get_bboxes,
)
SEED = 123
torch.manual_seed(SEED)

# Hyper parameter etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0 
NUM_WORKERS = 2
EPOCH = 100
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_PATH = "overfit.pth.tar"

class Compose(object):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
    def __call__(self, img, bboxes) :
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

def train_fn(model, loss_fn, optimizer, train_loader):
    # for each item in dataset, (img_tensor, label_matrix)
    # img_tensor: (3, 448, 448)
    # label_matrix: (7, 7, 25) where 25 = 20 + 5 * 1
    
    mean_loss = []
    for idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss.append(loss)
    
    print(f"Mean loss of each batch was {sum(mean_loss) / len(mean_loss)}")
    
def main():
    model = YOLOv1().to(DEVICE)
    loss_fn = YoloLoss()
    optimizer = optim.Adam(model.parameters(),LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    transform = Compose((transforms.Resize((448, 448)), transforms.ToTensor()))
    if LOAD_MODEL:
        load_checkpoint(LOAD_MODEL_PATH, model, optimizer)
    
    trian_dataset = VOCDataset(
        "data/images",
        "data/labels",
        "data/8examples.csv",
        transform=transform
    )
    
    
    test_dataset = VOCDataset(
        "data/images",
        "data/labels",
        "data/test.csv",
        transform=transform
    )
    
    train_dataloader = DataLoader(
        trian_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,       
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,       
    )
    
    for epoch in range(EPOCH):
        print(f"***************EPOCH: {epoch}*****************")
        pred_boxes, target_boxes = get_bboxes(
            train_dataloader, model
        )
        if epoch==50:
            print("epoch stop")
        mAP = mean_average_precision(
            pred_boxes, target_boxes
        )
        print(f"Train mAP: {mAP}")
        train_fn(model, loss_fn, optimizer, train_dataloader)

if __name__== "__main__":
    main()


