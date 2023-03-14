import os
import shutil

# make train_yolo and test_yolo to move yolo annotations into seperate folders

os.system('mkdir data/train_yolo')
os.system('mkdir data/test_yolo')

all_files = list(sorted(os.listdir("data/annotations_yolo/")))

for file in all_files:
    num = int(file[:2])
    original = "data/annotations_yolo/" + file
        
    
    if num < 12:
        target = "data/train_yolo/" + file
        shutil.copy(original, target)
    else:
        target = "data/test_yolo/" + file
        shutil.copy(original, target)
        
    