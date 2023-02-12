import os
import shutil
import sys

import ee
import geemap
import rasterio
import cv2
from PIL import Image

from osgeo import gdal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import date
from datetime import timedelta

import ast

import joblib

def image_downloader(place_coords, start_date, end_date):
    
    # input given as 'MM/DD/YYY'
    start = datetime.strptime(start_date, '%m/%d/%Y')
    end = datetime.strptime(end_date, '%m/%d/%Y')

    # create folders if not already created
    if not os.path.exists('gee_data'):
        os.mkdir('gee_data')
    if not os.path.exists('gee_data/full_img'):
        os.mkdir('gee_data/full_img')
        
    bbox = place_coords
    
    region = ee.Geometry.Polygon(bbox)

    collection = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(start_date,end_date).filterBounds(region)
    collection_list = collection.toList(collection.size())
    dates = geemap.image_dates(collection, date_format='YYYY-MM-dd').getInfo()
        
    for i, date in enumerate(dates[:]):
        
        image = ee.Image(collection_list.get(i)).select('VV')

        try:
            geemap.ee_export_image(image, filename = "gee_data/full_img/{}.tif".format(date), region = region)
        except Exception as e:
            print(e)

    print('Successfully Downloaded')
    return




def image_splitter(img_fp):
    """
    Takes the image filepath returns m x n array of the subimages
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
    
    img_height, img_width = img_array.shape
    # get next biggest multiple of 800
    new_height = img_height + (800 - img_height % 800)
    new_width = img_width + (800 - img_width % 800)
    
    reshaped = cv2.resize(img_array, (new_height, new_width))
    
    # for some reason have to rescale or else when saving image it will be dark
    rescaled = a_scaled = 255*(reshaped-reshaped.min())/(reshaped.max()-reshaped.min())

    # normalize between 0 and 1
    rescale_normalized = rescaled / 255
    
    # split image into subimages
        # will return array [m, n, 800, 800] where there are an m x n number of images with size 800x800 
    split = reshape_split(rescale_normalized, kernel_size=(800,800))
    return split, img_name