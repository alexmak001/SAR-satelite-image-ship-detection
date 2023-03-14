### Downloads data AND models

import os
import gdown
import shutil

# down load data, create new directory called data

# down load data, create new directory called data

# need to change this when make new data zip
# link of old dataset w missing data in test
#'https://drive.google.com/uc?id=1l5o0N59Ut4-ahZxYdFZa6vT_xytjuEjY'


os.system("mkdir data")


url = 'https://drive.google.com/uc?id=1fdp8nBcGBjOv6XoPMEJdZDdUaG1gmWxh&confirm=t'
output = 'data_download.zip'


# download and unzip
gdown.download(url, output, quiet=False)
os.system('unzip data_download.zip')


shutil.move("annotations/","data/")
shutil.move("annotations_yolo/","data//")
shutil.move("images/","data/")
shutil.move("main/","data/")

os.system("rm data_download.zip")



#make train_yolo and test_yolo to move yolo annotations into seperate folders

os.system('mkdir data/train')
os.system('mkdir data/test')

all_files = list(sorted(os.listdir("data/annotations_yolo/")))

for file in all_files:
    num = int(file[:2])
    original = "data/annotations_yolo/" + file
        
    
    if num < 11:
        target = "data/train/" + file
        shutil.copy(original, target)
    else:
        target = "data/test/" + file
        shutil.copy(original, target)


# download models
url = 'https://drive.google.com/uc?id=1oHU3HUvNPOw6H6sFsLAxdytNTtevcXei'
output = 'models_download.zip'

gdown.download(url, output, quiet=False)

os.system('unzip models_download.zip')
os.system("rm models_download.zip")

os.system("mv Models/retina300R2.pt src/models/")
os.system("mv Models/faster300ep.pt src/models/")
os.system("rmdir Models")
