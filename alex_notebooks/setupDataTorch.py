import os
import gdown
import shutil

# down load data, create new directory called data

# down load data, create new directory called data

# need to change this when make new data zip
# link of old dataset w missing data in test
# 'https://drive.google.com/uc?id=1l5o0N59Ut4-ahZxYdFZa6vT_xytjuEjY'
url = 'https://drive.google.com/uc?id=1fdp8nBcGBjOv6XoPMEJdZDdUaG1gmWxh&confirm=t'
output = 'data_download.zip'

gdown.download(url, output, quiet=False)
# make data folder
os.system('mkdir data')
# move zip file into 
os.system(r'mv data_download.zip data')

# change directory to data folder
os.chdir('data')

# unzip data download
os.system('unzip data_download.zip')
os.system("rm data_download.zip")
os.chdir('..')

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
        
    