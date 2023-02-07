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
import utils
import wandb
from tqdm import tqdm


# CUDA
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# WandB – Initialize a new run
wandb.init(entity="ship-detection", project="sar-ship-detection")

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
config.batch_size = 10         # input batch size for training (default: 64)
config.epochs = 300             # number of epochs to train (default: 10)
config.lr = 0.001               # learning rate (default: 0.01)
config.momentum = 0.9         # SGD momentum (default: 0.5) 
config.weight_decay = 0         # weight decay
config.seed = 42               # random seed (default: 42)
config.log_interval = 10     # how many batches to wait before logging training status
config.threshold = 0.5      # confidence threshold for an object to be considered to be detected

torch.manual_seed(config.seed)

class ShipDataset:
    def __init__(self, path, transform=None):
        self.path = path
        self.files = list(sorted(os.listdir("data/annotations_yolo/")))
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
        target = utils.generate_target(label_path)

        
        if self.transform:
            transformed = self.transform(image = image, bboxes = target['boxes'], labels = target['labels'])
            image = torch.Tensor(transformed['image'])
            target = {'boxes':torch.Tensor(transformed['bboxes']).reshape(-1,4), 'labels':torch.Tensor(transformed['labels'])}
        else:
            image = torch.tensor(image,dtype=torch.float32)
        
        image = torch.unsqueeze(image, dim=0)
            
        return image, target 

def main(model_name):

    print("Loading Data")
    # load the data
    dataset = ShipDataset(
        path = 'images/'
    )

    train_set, val_set = torch.utils.data.random_split(dataset, [1500,359],)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, collate_fn=utils.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size,collate_fn=utils.collate_fn)

    print("Loading Model")
    # load the model
    retina = torchvision.models.detection.retinanet_resnet50_fpn(num_classes = 2, pretrained=False, pretrained_backbone = True)

    num_epochs = config.epochs
    retina.to(device)
        
    # parameters
    params = [p for p in retina.parameters() if p.requires_grad] # select parameters that require gradient calculation
    
    # weight decay or nah
    if config.weight_decay > 0 :
        optimizer = torch.optim.SGD(params, lr=config.lr,
                                    momentum=config.momentum,weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=config.lr,
                                    momentum=config.momentum)

    print("Training Model")
    # train model
    len_dataloader = len(train_loader)
    all_loss = []
    
    for epoch in range(num_epochs):
        start = time.time()
        retina.train()

        i = 0    
        epoch_loss = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = retina(images, targets) 

            losses = sum(loss for loss in loss_dict.values()) 

            i += 1

            optimizer.zero_grad()
            losses.backward()

            torch.nn.utils.clip_grad_norm_(params,max_norm=2.0)
            optimizer.step()
            
            epoch_loss += losses 
        all_loss.append(epoch_loss)

        if epoch % config.log_interval == 0:
            precision, recall, AP, f1, mAP = utils.calc_test_stats(val_loader, retina, config.threshold, device)
            wandb.log({"epoch":epoch, "loss": epoch_loss, "precision": precision,"recall":recall,"AP":AP,"mAP":mAP,"f1":f1})
            print("Epoch #: {0}, Loss: {1}, Time: {2}".format(epoch, epoch_loss.item(),time.time() - start))
    
    print("Finished Training")
    print("Saving Model")
    # save model
    torch.save(retina.state_dict(), "models/"+model_name)
    torch.save(retina.state_dict(), os.path.join(wandb.run.dir, model_name))
    print("Done!")

if __name__ == '__main__':  
    model_name = sys.argv[1]
    main(model_name)