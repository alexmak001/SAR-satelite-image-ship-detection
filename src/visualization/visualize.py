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
from matplotlib import pyplot as plt
import wandb
from tqdm import tqdm
import rasterio

# CUDA
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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


model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 2  # 1 class (ship) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("src/models/faster300ep.pt"))
model.eval()
model.to(device)

feb_fp = "src/visualization/2020-02-04.tif"
april_fp = "src/visualization/2020-04-11.tif"

plot_large_image(model,feb_fp,0.7,"src/visualization/before_covid.jpg",device)

plot_large_image(model,april_fp,0.7,"src/visualization/after_covid.jpg",device)
