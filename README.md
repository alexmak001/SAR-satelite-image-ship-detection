# SAR-satelite-image-ship-detection
In the past, synthetic aperture radar (SAR) images did not have a high resolution, and required immense computational power to create. However, due to recent advances in remote sensing, SAR image quality has significantly increased and is becoming a hot topic in research. One application of these images is to detect container ships traveling through bodies of water. This paper uses the Large-Scale SAR Ship Detection Dataset-v1.0 (LS-SSDD-v1.0) which contains large scale SAR images from Sentinel 1 satellites that contain ships and their respective labels. These large images are cut into smaller pieces in order to train a variety of object detection models and deploy them on Sentinel-1 image data from Google Earth Engine. 

## Testing:
When running on DSMLP, be sure to use use the launch script
`launch-scipy-ml.sh -g 1 -i snng/sar_ship_detection` to launch a pod with a GPU. Otherwise the script will fail to run. 

To run the test, simply run python run.py test

## HOW TO RUN THE SHIP COUNTING SCRIPT:
[//]: <> (Have to figure out what to do about json key)
1. Once you have activated your DSMLP environment using the launch script above, please ensure you have the private json key downloaded. This will be used for initializing and authenticating Google Earth Engine. [website](https://developers.google.com/earth-engine/guides/service_account)
2. We will have to get the coordinates for the desired area of interest from the Google Earth Engine [website](https://code.earthengine.google.com/).
    - Note: If you do not have a Google Earth Engine account you will have to make one.
    - To get the coordinates, navigate on the map to your desired place of interest
    - Then draw a bounding box over your area of interest using the shape tool. (The shape tool is button with the gray square underneath the scripts panel. 
    ![tut1](https://user-images.githubusercontent.com/69220036/221438416-ca8513ea-412e-43c6-8a8e-5b87e30ac128.png)
    ![tut2](https://user-images.githubusercontent.com/69220036/221438475-eac5c729-4478-46bd-8691-88648845255a.png)
    - Once you get the desired vertices, the middle panel which is usually labeled new script will have a geometry variable and you expand that until you get the list of 5 vertices and those are the place coordinate values to pass into image downloader function.
  ![tut3](https://user-images.githubusercontent.com/69220036/221438515-9acf67df-450b-4f66-b4a7-deed39eb1013.png)
3. Once you have obtained your coordinates, run this command in terminal to start counting ships.
[//]: <> (Might have to change this command depending on how we implement shipcounter.py.)

`python -c from shipcounter.py import shipcounter(place_coords, start_date, end_date, del_images)`

where:
- `place_coords` are the coordinates from Google Earth Engine
- `start_date` is the desired start date in the format 'MM/DD/YYYY'
- `end_date` is the desired end date in the format 'MM/DD/YYYY'
- `del_images` set to `True` if you want to delete the images locally afterwards, `False` if not.
