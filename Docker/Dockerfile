# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools
# See https://github.com/ucsd-ets/datahub-docker-stack/wiki/Stable-Tag 
# for a list of the most current containers we maintain
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

RUN apt update

RUN apt-get -y install aria2 nmap traceroute
# RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update
# RUN apt-get install gdal-bin
# RUN apt-get install libgdal-dev
# RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
# RUN export C_INCLUDE_PATH=/usr/include/gdal

# 3) install packages using notebook user
USER jovyan

# RUN conda install -y scikit-learn

RUN pip install --no-cache-dir pandas geopandas earthengine-api geemap datetime numpy rasterio matplotlib folium gdown
COPY requirements.txt .
RUN pip install -r requirements.txt

# clone yolov5
RUN git clone https://github.com/ultralytics/yolov5

# clone yolov7
RUN git clone https://github.com/WongKinYiu/yolov7

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]