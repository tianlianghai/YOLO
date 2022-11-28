import torch
import pandas as pd
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, img_dir, label_dir, csv_dir, S=7, C=20, B=2, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.csv_dir = csv_dir
        self.S = S
        self.C = C
        self.B = B
        self.transform = transform
        self.annots = pd.read_csv(self.csv_dir)
    
    def __len__(self):
        return len(self.annots)
    
    def __getitem__(self, index):
        S = self.S
        C = self.C
        B = self.B
        
        annots = self.annots
        label_path = os.path.join(self.label_dir,
                         annots.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                box = [int (s) if int(float(s)) == float(s) else float(s)
                       for s in label.split()]
                boxes.append(box)
        # for transform
        boxes = torch.tensor(boxes)
        
        img = Image.open(os.path.join(self.img_dir, annots.iloc[index, 0]))
        # transform
        if self.transform:
            img, boxes = self.transform(img, boxes)
        
        label_matrix = torch.zeros(S, S, C + 5 * B)
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(S * y), int(S * x)
            cell_x, cell_y = S * x - j, S * y - i
            width_cell, height_cell = (
                width * S,
                height * S
            )
            
            if label_matrix[i, j, C] == 0:
                label_matrix[i, j, C] = 1
                label_matrix[i, j, class_label] = 1
                box_coordinates = torch.tensor(
                    [cell_x, cell_y, width_cell, height_cell]
                )
                label_matrix[i, j, C+1:C+5] = box_coordinates
        
        
        return img, label_matrix 