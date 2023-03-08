import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from bs4 import BeautifulSoup
from PIL import Image
import cv2
import numpy as np
import time
import torch
import torchvision
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import os
import random
from tqdm import tqdm
import rasterio
# helper function for dataset
def generate_box(obj):
    
    xmin = float(obj.find('xmin').text)
    ymin = float(obj.find('ymin').text)
    xmax = float(obj.find('xmax').text)
    ymax = float(obj.find('ymax').text)
    
    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    # only have ships
    return 1 

def generate_target(file): 
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, "xml")
        objects = soup.find_all("object")

        num_objs = len(objects)

        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))


        boxes = torch.as_tensor(boxes, dtype=torch.float32) 
        labels = torch.as_tensor(labels, dtype=torch.int64) 
        
        target = {}

        
        target["boxes"] = boxes
        target["labels"] = labels
        
        return target

def plot_image_from_output(img, annotation):
    
    img = img.permute(1,2,0)
    
    fig,ax = plt.subplots(1)
    ax.imshow(img,cmap="gray")
    
    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 0 :
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')
        
        elif annotation['labels'][idx] == 1 :
            
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
            
        else :
        
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='orange',facecolor='none')
        
        ax.add_patch(rect)

    plt.show()

def collate_fn(batch):
    return tuple(zip(*batch))


def plot_image_from_output2(img, annotation):
    
    img = img.cpu().permute(1,2,0)
    
    rects = []

    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 0 :
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        
        elif annotation['labels'][idx] == 1 :
            
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
            
        else :
        
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='orange',facecolor='none')

        rects.append(rect)

    return img, rects

def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : #select idx which meets the threshold
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]


    return preds


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i] # predict
        # pred_boxes = output['boxes']
        # pred_scores = output['scores']
        # pred_labels = output['labels']

        true_positives = torch.zeros(output['boxes'].shape[0])   
 
        annotations = targets[sample_i]  # actual
        target_labels = annotations['labels'] if len(annotations) else []
        if len(annotations):    # len(annotations) = 3
            detected_boxes = []
            target_boxes = annotations['boxes']

            for pred_i, (pred_box, pred_label) in enumerate(zip(output['boxes'], output['labels'])): # 예측값에 대해서..

                # If targets are found break
                if len(detected_boxes) == len(target_labels): # annotations -> target_labels
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)   # box_index : 실제 어떤 바운딩 박스랑 IoU 가 가장 높은지 index
                if iou >= iou_threshold and box_index not in detected_boxes: 
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index] 
        batch_metrics.append([true_positives, output['scores'], output['labels']])
    return batch_metrics

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = torch.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = torch.unique(target_cls)   

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = torch.cumsum(1 - tp[i],-1)
            tpc = torch.cumsum(tp[i],-1)

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = torch.tensor(np.array(p)), torch.tensor(np.array(r)), torch.tensor(np.array(ap))
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def calc_test_stats(val_loader, model, threshold, device):
    labels = []
    preds_adj_all = []
    annot_all = []

    # inference
    for im, annot in val_loader:
        im = list(img.to(device) for img in im)
        #annot = [{k: v.to(device) for k, v in t.items()} for t in annot]

        for t in annot:
            labels += t['labels']

        with torch.no_grad():
            preds_adj = make_prediction(model, im, threshold)
            preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
            preds_adj_all.append(preds_adj)
            annot_all.append(annot)

    # get metrics on validation set
    sample_metrics = []
    for batch_i in range(len(preds_adj_all)):
        sample_metrics += get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5) 

    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))] 
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, torch.tensor(labels))
    mAP = torch.mean(torch.tensor(AP,dtype=torch.float64))

    return precision, recall, AP, f1, mAP

def image_splitter(img_fp):
    """
    Takes the image filepath returns m x n array of the subimages
    pads with 0
    """
    #     img = gdal.Open(img_fp)
    #     img_array = img.GetRasterBand(1).ReadAsArray()
    # note have to add in rescaled helper fn

    # define splitting helper function
    # got this from https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
    def reshape_split(image, kernel_size):
        img_height, img_width = image.shape
        tile_height, tile_width = kernel_size
        
        tiled_array = image.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width)
        tiled_array = tiled_array.swapaxes(1,2)
        return tiled_array


    # get image name to return
    # splits full fp by \, then gets the YYYY-MM-DD_#.jpg, then cuts off .jpg 
    img_name = img_fp.split('\\')[-1][:-4]



    with rasterio.open(img_fp) as src:
        img_array = src.read()[0]
        #print(img_array.shape)
        
    # have to clip values to take away gray tint from image
    # and have to do this so that inshore offshore classifier works
    img_array = np.clip(img_array, -20, 0)
    
    img_height, img_width = img_array.shape
    # get next biggest multiple of 800
    # TODO: Make it closest multiple of 800 instead OR PAD WITH BLACK
    new_height = img_height + (800 - img_height % 800)
    new_width = img_width + (800 - img_width % 800)
    
    # calculate number of pixels to pad
    delta_w = new_width - img_width
    delta_h = new_height - img_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # pad image with 0
    image_pad = np.pad(img_array, ((top, bottom), (left, right)), mode='constant', constant_values=-20)
    
    # for some reason have to rescale or else when saving image it will be dark
    rescaled = 255*(image_pad-image_pad.min())/(image_pad.max()-image_pad.min())

    # normalize between 0 and 1
    # MAKE A BETTER RESCALING
    rescale_normalized = rescaled / 255
    
    # split image into subimages
        # will return array [m, n, 800, 800] where there are an m x n number of images with size 800x800 
    split = reshape_split(rescale_normalized, kernel_size=(800,800))
    
    return split, img_name

def plot_large_image(model,image_path,threshold,save_path,device):
    """
    model: pytorch 
    image_path: path tif file
    threshold: confidence threshold for model
    save_path: path to save plot (must be jpg), if none will not save
    device: device
    """

    # split image and save size for future
    split_image, _ = image_splitter(image_path)
    split_size = split_image.shape[:2]

    # flatten image and format to work with pytroch device
    flattened = np.reshape(split_image, (-1, split_image.shape[2], split_image.shape[3]))
    torch_split_img = np.array(flattened)
    torch_split_img = torch.tensor(torch_split_img,dtype=torch.float32)
    torch_split_img = torch.unsqueeze(torch_split_img, dim=0)
    torch_split_img = torch_split_img.permute(1,0,2,3)
    torch_split_img = torch_split_img.to(device)

    # predict bounding boxes
    pred = model(torch_split_img)
    # reshape to match image split
    pred = np.reshape(pred,split_size)


    # Create a 3x3 plot with a shared axis
#     figsize=(split_size[0], split_size[1])
    fig, axs = plt.subplots(split_size[0], split_size[1], sharex=True, sharey=True, figsize=(split_size[1] * 2, 
                                                                                             split_size[0] * 2))

    # Loop over the images and annotations and plot them on the corresponding subplot
    for i in range(split_size[0]):
        for j in range(split_size[1]):
            # Plot the image
            axs[i, j].imshow(split_image[i, j], cmap='gray', aspect='auto')
            
            # Remove the axis labels and ticks
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].axis('off')
            # Loop over the annotations and plot them as bounding boxes
            curPred = pred[i, j]
            for k in range(len(curPred["scores"])):
                # checks if above threshold
                if curPred["scores"][k] > threshold:
                    annotation = curPred['boxes'][k]
                    annotation = annotation.cpu().detach().numpy()
                    left, upper, right, lower = annotation
                    width, height = right - left, lower - upper
                    rect = plt.Rectangle((left, upper), width, height, linewidth=1, edgecolor='r', facecolor='none')
                    axs[i, j].add_patch(rect)

    # Remove the space between the images
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_path:
        plt.savefig(save_path,quality=100,dpi=500)
    # Show the plot
    plt.show()


def get_metric(model,datalaoder,threshold,device):
    """
    Not to be used with YOLO
    """

    t1 = time.time()
    labels = []
    preds_adj_all = []
    annot_all = []

    for im, annot in datalaoder:
        im = list(img.to(device) for img in im)
        #annot = [{k: v.to(device) for k, v in t.items()} for t in annot]

        for t in annot:
            labels += t['labels']

        with torch.no_grad():
            preds_adj = make_prediction(model, im, threshold)
            preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
            preds_adj_all.append(preds_adj)
            annot_all.append(annot)
            
    
    sample_metrics = []
    for batch_i in range(len(preds_adj_all)):
        sample_metrics += get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5) 
    
    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))] 
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, torch.tensor(labels))
    mAP = torch.mean(torch.tensor(AP,dtype=torch.float64))
    t2 = time.time()
    t3 = t2-t1
    result = "Precision: {0:.3f} \n Recall: {1:.3f} \n AP: {2:.3f} \n F1: {3:.3f} \n mAP {4:.3f} \n time: {5:.3f} \n threshold: {6:}".format(precision.item(), recall.item(), AP.item(), f1.item(),mAP.item(), t3,threshold)
    #print(result)
    print(threshold + "done")
    return [precision.item(), recall.item(), AP.item(), f1.item(),mAP.item(), t3, threshold]

def plot_ground_truth_and_predictions(model, data_loader, device,threshold, n,save_path):
    """
    For RetinaNet and Faster R-CNN
    model: model to predict
    data_loader: test_data loader to use
    device: the device to use
    threshold: confidence value to threshold a prediction
    n: random into scroll on images
    save_path: save path of images, must end in .jpg

    returns: none
    prints a random image from data_loader with both its ground truth and the models predictions
    """
    # Get a batch of validation data
    data_loader = iter(data_loader)
    #images, targets = next(data_loader)
    images, targets = next(data_loader)
    
    for i in range(n):
        images, targets = next(data_loader)

    images = images[0]
    # Send the images and targets to the device
    images = images.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    images = torch.unsqueeze(images, dim=0)
    # Run the model on the images
    outputs = model(images)


    # Loop over the images in the batch
    for i in range(len(images)):
        # Get the image, target boxes, and predicted boxes for this image
        print(images[i].shape)
        image = images[i].cpu().permute(1, 2, 0).numpy()
        print(image.shape)
        target_boxes = targets[i]['boxes'].cpu().numpy()

        # grabs all ids that are above the threshold value
        idx_list = []
        for idx, score in enumerate(outputs[0]['scores']) :
            if score > threshold : #select idx which meets the threshold
                idx_list.append(idx)

        predicted_boxes = outputs[i]['boxes'][idx_list].detach().cpu().numpy()

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the ground truth boxes on the first subplot
        axs[0].imshow(image,cmap="gray")
        for box in target_boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=2, edgecolor='g', facecolor='none')
            axs[0].add_patch(rect)
        axs[0].set_title('Ground Truth')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        # Plot the predicted boxes on the second subplot
        axs[1].imshow(image,cmap="gray")
        for box in predicted_boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=2, edgecolor='r', facecolor='none')
            axs[1].add_patch(rect)
        axs[1].set_title('Predictions')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        # Show the plot
        if save_path:
            plt.savefig(save_path,quality=100,dpi=500)
        plt.show()

def plot_ground_truth_and_predictions_YOLO(model, data_loader, device,threshold, n,save_path):
    """
    For Yolo Model only!
    model: model to predict
    data_loader: test_data loader to use
    device: the device to use
    threshold: confidence value to threshold a prediction
    n: random into scroll on images
    save_path: save path of images, must end in .jpg

    returns: none
    prints a random image from data_loader with both its ground truth and the models predictions
    """
    # Get a batch of validation data

    data_loader = iter(data_loader)
    #images, targets = next(data_loader)
    images, targets = next(data_loader)
    
    for i in range(n):
        images, targets = next(data_loader)

    # image on cpu for yolo
    images = torch.squeeze(images[0],0).cpu().numpy()

    # images = images[0]
    # # Send the images and targets to the device
    # #images = images.to(device)
    # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # images = torch.unsqueeze(images, dim=0)
    # Run the model on the images
    outputs = model(images)

    # YOLO conversion
    outputs = outputs.pred[0]

    # Create a dictionary with keys 'boxes', 'labels', and 'scores'
    outputs = {
        'boxes': outputs[:, :4],  # only take the first four columns of the tensor
        'labels': outputs[:, 5].long()+1,
        'scores': outputs[:, 4]
    }
    outputs = [outputs]

    # Loop over the images in the batch
    for i in range(len(images)):
        # Get the image, target boxes, and predicted boxes for this image
        # make image 800, 800, 1
        image = images[ :, :, np.newaxis]
        target_boxes = targets[i]['boxes'].cpu().numpy()

        # grabs all ids that are above the threshold value
        idx_list = []
        for idx, score in enumerate(outputs[0]['scores']) :
            if score > threshold : #select idx which meets the threshold
                idx_list.append(idx)

        predicted_boxes = outputs[i]['boxes'][idx_list].detach().cpu().numpy()

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the ground truth boxes on the first subplot
        axs[0].imshow(image,cmap="gray")
        for box in target_boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=2, edgecolor='g', facecolor='none')
            axs[0].add_patch(rect)
        axs[0].set_title('Ground Truth')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        # Plot the predicted boxes on the second subplot
        axs[1].imshow(image,cmap="gray")
        for box in predicted_boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=2, edgecolor='r', facecolor='none')
            axs[1].add_patch(rect)
        axs[1].set_title('Predictions')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        # Show the plot
        
        if save_path:
            plt.savefig(save_path,quality=100,dpi=500)
        plt.show()

def make_prediction_yolo(model, img, threshold):
    model.eval()
    preds = model(img)
    
    preds = preds.pred[0]

    # Create a dictionary with keys 'boxes', 'labels', and 'scores'
    preds = {
        'boxes': preds[:, :4],  # only take the first four columns of the tensor
        'labels': preds[:, 5].long()+1,
        'scores': preds[:, 4]
    }
    preds = [preds]
    
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : #select idx which meets the threshold
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]


    return preds

def get_metric_YOLO(model, dataloader, threshold):
    t1 = time.time()
    labels = []
    preds_adj_all = []
    annot_all = []

    for im, annot in dataloader:
        
        for t in annot:
            
            labels += t['labels']
        
        with torch.no_grad():
            ### make a new function for yolo, need to move to cpu first
            # will slow down time significantly
            im = torch.squeeze(im[0],0).cpu().numpy()
            preds_adj = make_prediction_yolo(model, im, threshold)
            preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
            preds_adj_all.append(preds_adj)
            annot_all.append(annot)
    
    sample_metrics = []
    for batch_i in range(len(preds_adj_all)):
        sample_metrics += get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5) 
    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))] 
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, torch.tensor(labels))
    mAP = torch.mean(torch.tensor(AP,dtype=torch.float64))
    t2 = time.time()
    t3 = t2-t1
    result = "Precision: {0:.3f} \n Recall: {1:.3f} \n AP: {2:.3f} \n F1: {3:.3f} \n mAP {4:.3f} \n time: {5:.3f} \n threshold: {6:}".format(precision.item(), recall.item(), AP.item(), f1.item(),mAP.item(), t3,threshold)
    print(result)
    return [precision.item(), recall.item(), AP.item(), f1.item(),mAP.item(), t3, threshold]

def read_file(filename):
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines

# array of all test annotation with ships 
allShipsTest = os.listdir("test_yolo/")
allShipsTest = [ship[:-4] for ship in allShipsTest]
# array of all inshore annotations
allInshore = read_file("main/test_inshore.txt")
# array of all offshore annotations
allOffshore = read_file("main/test_offshore.txt")
set1 = set(allShipsTest)
set2 = set(allInshore)
set3 = set(allOffshore)

inshoreShipTest = list(set1.intersection(set2))
offshoreShipTest = list(set1.intersection(set3))

class ShipDataset:
    def __init__(self, path, transform=None):
        self.path = path
        if path == "no_ship":
            self.files = "chui"
        elif path == "test":
            self.files = allShipsTest
        elif path == "in":
            self.files = inshoreShipTest
        elif path == "off":
            self.files = offshoreShipTest
        else:
            self.files = list(sorted(os.listdir(path)))
        
        self.transform = transform
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.path in ["no_ship","test","in","off"]:
            file_image = self.files[idx] + '.jpg'
            file_label = self.files[idx] + '.xml'
        else:
            file_image = self.files[idx][:-3] + 'jpg'
            file_label = self.files[idx][:-3] + 'xml'

        img_path = os.path.join("images/", file_image)
        label_path = os.path.join("annotations/", file_label)
        #print(img_path,label_path)
        
        # Read an image with OpenCV, gray scale
        image = cv2.imread(img_path,0)
        
        image = image/255.0
        target = generate_target(label_path)

        
        if self.transform:
            transformed = self.transform(image = image, bboxes = target['boxes'], labels = target['labels'])
            image = torch.Tensor(transformed['image'])
            target = {'boxes':torch.Tensor(transformed['bboxes']).reshape(-1,4), 'labels':torch.Tensor(transformed['labels'])}
        else:
            image = torch.tensor(image,dtype=torch.float32)
        
        image = torch.unsqueeze(image, dim=0)
            
        return image, target 

class ShipDataset_YOLO:
    def __init__(self, path, transform=None):
        self.path = path
        if path == "no_ship":
            self.files = "chui"
        elif path == "test":
            self.files = allShipsTest
        elif path == "in":
            self.files = inshoreShipTest
        elif path == "off":
            self.files = offshoreShipTest
        else:
            self.files = list(sorted(os.listdir(path)))
        
        self.transform = transform
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.path in ["no_ship","test","in","off"]:
            file_image = self.files[idx] + '.jpg'
            file_label = self.files[idx] + '.xml'
        else:
            file_image = self.files[idx][:-3] + 'jpg'
            file_label = self.files[idx][:-3] + 'xml'

        img_path = os.path.join("images/", file_image)
        label_path = os.path.join("annotations/", file_label)
        #print(img_path,label_path)
        
        # Read an image with OpenCV, gray scale
        image = cv2.imread(img_path,0)
        
        image = image#/255.0
        target = generate_target(label_path)

        
        if self.transform:
            transformed = self.transform(image = image, bboxes = target['boxes'], labels = target['labels'])
            image = torch.Tensor(transformed['image'])
            target = {'boxes':torch.Tensor(transformed['bboxes']).reshape(-1,4), 'labels':torch.Tensor(transformed['labels'])}
        else:
            image = torch.tensor(image,dtype=torch.float32)
        
        image = torch.unsqueeze(image, dim=0)
            
        return image, target 

