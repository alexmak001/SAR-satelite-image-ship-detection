import os
import shutil
import sys

import ee
import geemap
import rasterio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import date
from datetime import timedelta

import ast

"""
This script is used to downlaod Sentinel-1 SAR satellite images from Google Earth Engine using the API.
When running the script, pass your list of coordinates, start date, and end date as the targets. Coordinates are acquired from Google Earth Engine.
Follow this format for running the script:
python gee_download.py '[(,),(,),(,),(,),(,)]' 'YYYY MM DD' 'YYYY MM DD'
    The coordinates should be passed as a list of tupes encased in quotation marks.
    The year, month, and date of the start and end dates should separated by spaces and encased in quotation marks as well (see above).
"""



def main(targets):
    # the given target is the coordinates,
    # print(targets)  
    # print(type(ast.literal_eval(targets[0])))
    coords = ast.literal_eval(targets[0])
    
    # input given as 'YYYY MM DD'
    start_str = str.split(targets[1])
    start_year = int(start_str[0])
    start_month = int(start_str[1])
    start_day = int(start_str[2])

    end_str = str.split(targets[2])
    end_year = int(end_str[0])
    end_month = int(end_str[1])
    end_day = int(end_str[2])

    start = datetime(start_year, start_month, start_day)
    end = datetime(end_year, end_month, end_day)

    # create folders if not already created
    if not os.path.exists('gee_data'):
        os.mkdir('gee_data')
    if not os.path.exists('gee_data/full_img'):
        os.mkdir('gee_data/full_img')

    # initialize Earth Engine
    service_account = 'snng-download@sar-ship-detection.iam.gserviceaccount.com'
    credentials = ee.ServiceAccountCredentials(service_account, 'gee_download_key/sar-ship-detection-fb527bcf2a6d.json')
    ee.Initialize(credentials)

    def getImages(place_coords, start_date, end_date):
        """
        Given a set of coordinates, start date, and end date, download images from Google Earth Engine into data folder

        place_coords: [(,), (,), (,), (,), (,)]
        start_date: datetime object
        end_date: datetime object
        """

        # bbox = [(-118.32027994564326,33.64246038322455),(-118.07789408138545,33.64246038322455),(-118.07789408138545,33.78867573774964),(-118.32027994564326,33.78867573774964),(-118.32027994564326,33.64246038322455)]
        bbox = place_coords
        
        # define specificed region
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
    
    getImages(coords, start, end)
        


if __name__ == '__main__':
    #
    targets = sys.argv[1:]
    main(targets)