import os
import shutil
import sys
import ee
import geemap
import rasterio
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
from datetime import timedelta
import joblib
import numpy as np
import ast
import joblib
import torch
import torchvision

#from osgeo import gdal

# change these values if needed
threshold = 0.5

# load models in 

# load in predictor
clf = joblib.load("sean_notebooks/inshore_offshore_clf_normal_model.pkl")

# TODO: move  model to device


# load in faster r cnn
faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 2  # 1 class (ship) + background
# get number of input features for the classifier
in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
faster_rcnn.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
faster_rcnn.load_state_dict(torch.load("alex_notebooks/models/faster300ep.pt"))
faster_rcnn.eval()

# initialize Earth Engine
service_account = 'snng-download@sar-ship-detection.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'alex_notebooks/models/sar-ship-detection-fb527bcf2a6d.json')
ee.Initialize(credentials)



def ship_counter(place_coords, start_date, end_date, del_images):

    path = "gee_data/"

    # downloads all images to a file called gee_data
    print("Downloading Images...")
    dates = image_downloader(place_coords, start_date, end_date,path)
    print("Finished Downloading Images")
    
    fnames = os.listdir(path)

    totalShip_count = []

    print("Counting Ships")
    for file in fnames:
        file_fp = path+file

        # splits each image into an m xn array of 800x800
        split_img, img_name = image_splitter(file_fp)
        
        # TODO: Convert it to a 1d array of images
        # flatten split array into a [m*n, 800, 800] array
        flattened = np.reshape(split_img, (-1, split_img.shape[2], split_img.shape[3]))


        m = flattened.shape[0]
        
        # counts number of ships in each sub image
        subImageCount = 0
        
        
        for i in range(m):
            # get cur image
            curImg = flattened[i]

            # classify to be inshore(0) or offshore (1)
            offshore = inshore_offshore_classifier(curImg) == 1

            print(img_name,curImg.shape, offshore)

            # need to format image for pytorch 
            curImg = torch.tensor(curImg,dtype=torch.float32)
            curImg = torch.unsqueeze(curImg, dim=0)
            curImg = [curImg]

            if offshore:
                numShip = detect_ships_inshore(curImg)
            else:
                numShip = detect_ships_inshore(curImg)

            subImageCount += numShip
        
        print(file,subImageCount.item())

        totalShip_count.append(subImageCount.item())

    
    # delete images locally
    if del_images:
        shutil.rmtree(path)
        print("file deleted")
    
    

    return dates, totalShip_count



def image_downloader(place_coords, start_date, end_date, path):
    
    # input given as 'MM/DD/YYYY'
    start_date = datetime.strptime(start_date, '%m/%d/%Y')
    end_date = datetime.strptime(end_date, '%m/%d/%Y')

    
    # create folders if not already created
    if not os.path.exists(path):
        os.mkdir(path)
    
        
    bbox = place_coords
    
    region = ee.Geometry.Polygon(bbox)

    collection = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(start_date,end_date).filterBounds(region)
    collection_list = collection.toList(collection.size())
    dates = geemap.image_dates(collection, date_format='YYYY-MM-dd').getInfo()
        
    for i, date in enumerate(dates[:]):
        
        image = ee.Image(collection_list.get(i)).select('VV')

        try:
            geemap.ee_export_image(image, filename = path+"{}.tif".format(date), region = region)
        except Exception as e:
            print(e)

    print('Successfully Downloaded')
    
    return dates



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
        
    # have to clip values to take away gray tint from image
    # and have to do this so that inshore offshore classifier works
    img_array = np.clip(img_array, -20, 0)
    
    img_height, img_width = img_array.shape
    # get next biggest multiple of 800
    new_height = img_height + (800 - img_height % 800)
    new_width = img_width + (800 - img_width % 800)
    
    reshaped = cv2.resize(img_array, (new_height, new_width))
    
    # for some reason have to rescale or else when saving image it will be dark
    rescaled = a_scaled = 255*(reshaped-reshaped.min())/(reshaped.max()-reshaped.min())

    # normalize between 0 and 1
    # TODO MAKE A BETTER RESCALING
    rescale_normalized = rescaled / 255
    
    # split image into subimages
        # will return array [m, n, 800, 800] where there are an m x n number of images with size 800x800 
    split = reshape_split(rescale_normalized, kernel_size=(800,800))
    
    return split, img_name




def inshore_offshore_classifier(img):
    """
    Takes in an image and classifies it as either offshore(1) or inshore(0)
    """

    
    img_vals = np.copy(img)
    img_50 = np.percentile(img_vals,50)
    img_80 = np.percentile(img_vals,80)
    img_90 = np.percentile(img_vals,90)
    img_30 = np.percentile(img_vals,30)
    
    features = np.array([[img_50, img_80, img_90, img_30]])
    return clf.predict(features)[0]


def detect_ships_inshore(image):

    prediction = faster_rcnn(image)
    return sum(prediction[0]["scores"]>threshold)


def main():

    place_coords = [(-118.32027994564326,33.64246038322455),(-118.07789408138545,33.64246038322455),(-118.07789408138545,33.78867573774964),(-118.32027994564326,33.78867573774964),(-118.32027994564326,33.64246038322455)]
    # datetime(start_year, start_month, start_day)
    # string should be formatted as "YEAR MONTH DAY"
    start_date = "01/01/2020"
    end_date = "01/12/2020"
    del_images = False

    dates, totalShip_count = ship_counter(place_coords, start_date, end_date, del_images)

    print(dates)
    print(totalShip_count)

    return 

if __name__ == '__main__':
    
    main()