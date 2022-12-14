# SAR-satelite-image-ship-detection
In the past, synthetic aperture radar (SAR) images did not have a high resolution, and required immense computational power to create. However, due to recent advances in computer vision, SAR image quality has significantly increased and is becoming a hot topic in research. One application of these images is to detect container ships traveling through bodies of water. This paper uses the Large-Scale SAR Ship Detection Dataset-v1.0 (LS-SSDD-v1.0) which contains large scale SAR images from Sentinel 1 satellites that contain ships and their respective labels. These large images are cut into smaller pieces in order to train a variety of object detection models including Detectron2, and YOLOv7. Then, we deploy the model by scraping SAR images from Google Earth Engine from anywhere in the world and modeling the number of cargo containers in the region.

## Testing:
When running on DSMLP, be sure to use use the launch script
`launch-scipy-ml.sh -g 1 -i snng/sar_ship_detection` to launch a pod with a GPU. Otherwise the script will fail to run. 
