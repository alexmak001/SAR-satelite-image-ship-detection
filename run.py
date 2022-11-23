# dummy run file for now

import sys
import os
import json
import shutil

def main(targets):


    if 'test' in targets:
        
        # clone yolov7
        os.system("git clone https://github.com/WongKinYiu/yolov7")

        # fixes borkbne loss.py file
        os.system("rm yolov7/utils/loss.py")
        shutil.copy("datahelper/loss.py", "yolov7/utils/")

        # make directory copy test data x3
        os.system("mkdir yolov7/train/")
#         os.system("mkdir yolov7/train/images")
#         os.system("mkdir yolov7/train/annotations")

        os.system("cp -r test/test_data/images/ yolov7/train/images")
        os.system("cp -r test/test_data/annotations_yolo/ yolov7/train/annotations")

        os.system("mkdir yolov7/test/")
#         os.system("mkdir yolov7/test/images")
#         os.system("mkdir yolov7/test/annotations")

        os.system("cp -r test/test_data/images/ yolov7/test/images")
        os.system("cp -r test/test_data/annotations_yolo/ yolov7/test/annotations")

        os.system("mkdir yolov7/val/")
#         os.system("mkdir yolov7/val/images")
#         os.system("mkdir yolov7/val/annotations")

        os.system("cp -r test/test_data/images/ yolov7/val/images")
        os.system("cp -r test/test_data/annotations_yolo/ yolov7/val/annotations")

        shutil.copy("datahelper/sar_ship_dataset.yaml","yolov7/")

        # we run our YOLO training tiny model for 1 epoch
        print("Data downloaded and moved successfully!")
        print("Starting model training:")

        

        return True


if __name__ == '__main__':
    # run via:
    # python run.py
    targets = sys.argv[1:]
    main(targets)