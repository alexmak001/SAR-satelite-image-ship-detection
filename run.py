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

        os.system("cp -r test/test_data/images/ yolov7/train/images")
        os.system("cp -r test/test_data/annotations_yolo/ yolov7/train/labels")

        os.system("mkdir yolov7/test/")

        os.system("cp -r test/test_data/images/ yolov7/test/images")
        os.system("cp -r test/test_data/annotations_yolo/ yolov7/test/labels")

        os.system("mkdir yolov7/val/")

        os.system("cp -r test/test_data/images/ yolov7/val/images")
        os.system("cp -r test/test_data/annotations_yolo/ yolov7/val/lables")

        shutil.copy("datahelper/sar_dataset.yaml","yolov7/")

        # we run our YOLO training tiny model for 1 epoch
        print("Data downloaded and moved successfully!")
        print("Starting model training:")
        
        os.chdir('yolov7')
        os.system('python train.py --img 640 --batch 5 --epochs 1 --data sar_dataset.yaml --weights "yolov7.pt" --project sar-ship-detection --bbox_interval 1 --save_period 1')
        
        os.chdir('..')
        
        # output message saying where model is saved
        print('Model saved at: yolov7/sar-ship-detection')
        

        

        return True


if __name__ == '__main__':
    # run via:
    # python run.py
    targets = sys.argv[1:]
    main(targets)