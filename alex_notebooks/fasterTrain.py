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
config.epochs = 300            # number of epochs to train (default: 10)
config.lr = 0.001              # learning rate (default: 0.01)
config.momentum = 0.7          # SGD momentum (default: 0.5) 
config.weight_decay = 0.01     # weight decay
config.seed = 42               # random seed (default: 42)
config.log_interval = 5        # how many batches to wait before logging training status
config.threshold = 0.3         # confidence threshold for an object to be considered to be detected

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
    # load a model pre-trained on COCO
    fasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (ship) + background
    # get number of input features for the classifier
    in_features = fasterRCNN.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    fasterRCNN.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    num_epochs = config.epochs
    fasterRCNN.to(device)
        
    # parameters
    params = [p for p in fasterRCNN.parameters() if p.requires_grad] # select parameters that require gradient calculation
    optimizer = torch.optim.SGD(params, lr=config.lr,
                                    momentum=config.momentum)

    print("Training Model")
    # train model
    len_dataloader = len(train_loader)
    all_loss = []
    
    for epoch in range(num_epochs):
        start = time.time()
        fasterRCNN.train()

        i = 0    
        epoch_loss = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = fasterRCNN(images, targets) 

            losses = sum(loss for loss in loss_dict.values()) 

            i += 1

            optimizer.zero_grad()
            losses.backward()

            torch.nn.utils.clip_grad_norm_(params,max_norm=2.0)
            optimizer.step()
            
            epoch_loss += losses 
        all_loss.append(epoch_loss)

        if epoch % config.log_interval == 0:
            wandb.log({"loss": epoch_loss})
            print("Epoch #: {0}, Loss: {1}, Time: {2}".format(epoch, epoch_loss.item(),time.time() - start))
    
    print("Finished Training")
    print("Saving Model")
    # save model
    torch.save(fasterRCNN.state_dict(), "models/"+model_name)
    torch.save(fasterRCNN.state_dict(), os.path.join(wandb.run.dir, model_name))

    print("Calculating Test Statistics")
    labels = []
    preds_adj_all = []
    annot_all = []

    # inference
    for im, annot in tqdm(val_loader, position = 0, leave = True):
        im = list(img.to(device) for img in im)
        #annot = [{k: v.to(device) for k, v in t.items()} for t in annot]

        for t in annot:
            labels += t['labels']

        with torch.no_grad():
            preds_adj = utils.make_prediction(fasterRCNN, im, config.threshold)
            preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
            preds_adj_all.append(preds_adj)
            annot_all.append(annot)

    # get metrics on validation set
    sample_metrics = []
    for batch_i in range(len(preds_adj_all)):
        sample_metrics += utils.get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5) 

    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))] 
    precision, recall, AP, f1, ap_class = utils.ap_per_class(true_positives, pred_scores, pred_labels, torch.tensor(labels))
    mAP = torch.mean(torch.tensor(AP,dtype=torch.float64))

    wandb.log({"precision": precision,"recall":recall,"AP":AP,"mAP":mAP,"f1":f1})
    print("Done!")

if __name__ == '__main__':  
    model_name = sys.argv[1]
    main(model_name)