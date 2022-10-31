# dummy run file for now

import sys
import os
import json


def main(targets):


    if 'test' in targets:
        try:
            import ee

            import geemap
            import folium
            import rasterio
            from matplotlib import pyplot
            from osgeo import gdal
            print("All Imports Work!")

        except:
            print("Import Failed")
    return


if __name__ == '__main__':
    # run via:
    # python run.py
    targets = sys.argv[1:]
    main(targets)