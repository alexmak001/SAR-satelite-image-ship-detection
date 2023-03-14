import os
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt

import geemap
import rasterio
import cv2
from PIL import Image

from osgeo import gdal

full_img_fp = os.path.abspath('gee_data/full_img')

# crete subimg folder
if not os.path.exists('gee_data/sub_img'):
    os.mkdir('gee_data/sub_img')
sub_img_fp = os.path.abspath('gee_data/sub_img')

fnames = os.listdir(full_img_fp)

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

for file in fnames:
    img = gdal.Open(full_img_fp + '/{}'.format(file))
    img_array = img.GetRasterBand(1).ReadAsArray()
    img_height, img_width = img_array.shape

    # get next biggest multiple of 800
    new_height = img_height + (800 - img_height % 800)
    new_width = img_width + (800 - img_width % 800)

    # resize
    reshaped = cv2.resize(img_array, (new_height, new_width))

    # for some reason have to rescale or else when saving image it will be dark
    rescaled = a_scaled = 255*(reshaped-reshaped.min())/(reshaped.max()-reshaped.min())

    # split image into subimages
        # will return array [m, n, 800, 800] where there are an m x n number of images with size 800x800 
    split = reshape_split(rescaled, kernel_size=(800,800))

    m = split.shape[0]
    n = split.shape[1]
    date_fname = file.split('.')[0]

    for i in range(m):
        for j in range(n):
            im = split[i][j]
            im = Image.fromarray(im)
            im = im.convert("L")
            im.save(sub_img_fp + '/' + date_fname + '_' + str(i + j + 1) + '.jpg')
            



