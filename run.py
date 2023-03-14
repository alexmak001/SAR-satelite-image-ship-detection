
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
import ast

import joblib

def main(targets):


    if 'test' in targets:
        
        print("Test")

    if "data" in targets:
        os.system("python3 src/data/make_dataset.py ")

    if "train_ret" in targets:
        os.system("python3 src/models/train_retina_model.py ")

    if "train_faster" in targets:
        os.system("python3 src/models/train_fasterRCNN_model.py ")

    if "predict" in targets:
        os.system("python3 src/models/predict.py ")

    if "viz" in targets:
        os.system("python3 src/visualization/visualize.py ")

if __name__ == '__main__':
    # run via:
    # python run.py
    targets = sys.argv[1:]
    main(targets)