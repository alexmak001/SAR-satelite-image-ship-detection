# dummy run file for now

# import sys
# import os
# import json
# import shutil

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
        print("a")


if __name__ == '__main__':
    # run via:
    # python run.py
    targets = sys.argv[1:]
    main(targets)