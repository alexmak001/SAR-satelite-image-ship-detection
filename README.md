# Maritime Ship Detection Using Synthetic Aperture Radar Satellite Imagery
Satellites are being launched into space at an exponential rate and are able to produce high quality images in relatively short intervals of time on any part of Earth. The amount of data and types of it are also increasing significantly and in this paper we specifically use Synthetic Aperture Radar (SAR) satellite imagery in order to detect ships traveling through bodies of water. We created a ship counting tool that intakes a start date, end date, and an area of interest and returns the number of ships for each day between the two dates. The images are first classified into offshore or inshore and a separate object detection algorithm counts the number of ships per image. The classifier and object detection networks are trained using the Large-Scale SAR Ship Detection Dataset-v1.0 (LS-SSDD-v1.0) and deployed on Google Earth Engine.

## Testing:
When running on DSMLP, be sure to use use the launch script
`launch-scipy-ml.sh -g 1 -i snng/sar_ship_detection` to launch a pod with a GPU. Otherwise the script will fail to run. 

To run the test, simply run python run.py test

## run.py file
Using the "data" target, it downaloads and formats the dataset locally. This also downloads the models as well. 
The "train_ret" target will start the training of the RetinaNet model, which requires the data to be loaded. Similarly, the "train_faster" will begin to train the Faster R-CNN model. The hyperparameters can be configured in the src/models/ folder for each of the models.
The "predict" target uses the model to predict on all of the test data and returns the key metrics for both models. The "viz" target causes the models to predict the bounding boxes on the tif files saved in the src/visualization folder. It then saves the resulting image with bounding boxes in the same folder as a jpg.

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
