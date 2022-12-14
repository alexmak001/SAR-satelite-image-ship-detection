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


# move xml script inside and run it to create 

#shutil.copy("datahelper/xmltoyolo.py","data/")

#os.system("python3 data/xmltoyolo.py")

# split data into test and train accordingly

os.mkdir("data/train")
os.mkdir("data/train/images")
os.mkdir("data/train/labels")


# move all training sub images to one folder

with open("data/main/train.txt") as f:
    while True:
        line =  f.readline()
        line = line.replace("\n","")
        filename = line + ".jpg"

        original = "data/images/" + filename
        
        target = "data/train/images/" + filename

        try:
            shutil.move(original, target)
        except:
            print(filename + " NOT FOUND")

        if not line:
            break


# move all training sub annotations to one folder

with open("data/main/train.txt") as f:
    while True:
        line =  f.readline()
        line = line.replace("\n","")
        filename = line + ".txt"

        original = "data/annotations_yolo/" + filename
        
        target = "data/train/labels/" + filename

        try:
            shutil.move(original, target)
        except:
            print(filename + " NOT FOUND")

        if not line:
            break
        


os.mkdir("data/test")
os.mkdir("data/test/images")
os.mkdir("data/test/labels")



# move all test sub images to one folder

with open("data/main/test.txt") as f:
    while True:
        line =  f.readline()
        line = line.replace("\n","")
        filename = line + ".jpg"

        original = "data/images/" + filename
        
        target = "data/test/images/" + filename

        try:
            shutil.move(original, target)
        except:
            print(filename + " NOT FOUND")

        if not line:
            break


# move all test sub annotations to one folder

with open("data/main/test.txt") as f:
    while True:
        line =  f.readline()
        line = line.replace("\n","")
        filename = line + ".txt"

        original = "data/annotations_yolo/" + filename
        
        target = "data/test/labels/" + filename

        try:
            shutil.move(original, target)
        except:
            print(filename + " NOT FOUND")

        if not line:
            break
        

# download yolov7, copy data and yaml, edit utils/loss.py

os.system("git clone https://github.com/WongKinYiu/yolov7")

os.system("rm yolov7/utils/loss.py")
shutil.copy("datahelper/loss.py", "yolov7/utils/")

os.system("cp -r data/train/ yolov7/train")
os.system("cp -r data/test/ yolov7/test")

os.system("cp -r yolov7/test yolov7/val")

shutil.copy("datahelper/sar_dataset.yaml","yolov7/")
