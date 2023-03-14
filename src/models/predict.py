import os
import glob
from bs4 import BeautifulSoup
import sys 
from PIL import Image
import cv2
import numpy as np
import time
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import os
import random
import utility
import wandb
from tqdm import tqdm


# CUDA
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ShipDataset:
    def __init__(self, path, transform=None):
        self.path = path
        self.files = list(sorted(os.listdir(path)))
        self.transform = transform
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_image = self.files[idx][:-3] + 'jpg'
        file_label = self.files[idx][:-3] + 'xml'

        img_path = os.path.join("data/images/", file_image)
        label_path = os.path.join("data/annotations/", file_label)
        
        
        # Read an image with OpenCV, gray scale
        image = cv2.imread(img_path,0)
        
        image = image/255.0
        target = utility.generate_target(label_path)

        
        if self.transform:
            transformed = self.transform(image = image, bboxes = target['boxes'], labels = target['labels'])
            image = torch.Tensor(transformed['image'])
            target = {'boxes':torch.Tensor(transformed['bboxes']).reshape(-1,4), 'labels':torch.Tensor(transformed['labels'])}
        else:
            image = torch.tensor(image,dtype=torch.float32)
        
        image = torch.unsqueeze(image, dim=0)
            
        return image, target 
    
val_dataset = ShipDataset(
    path = 'data/test/'
)
   

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,collate_fn=utility.collate_fn)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 2  # 1 class (ship) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("src/models/faster300ep.pt"))
model.eval()
model.to(device)

# Print Faster R-CNN metrics
print("Faster R-CNN Metrics:")
utility.get_metric(model,val_loader,0.7,device)

print("\n \n")


# Retina
model = torchvision.models.detection.retinanet_resnet50_fpn(num_classes=2)
model.load_state_dict(torch.load("src/models/retina300R2.pt"))
model.eval()
model.to(device)

print("RetinaNet Metrics:")
utility.get_metric(model,val_loader,0.8,device)