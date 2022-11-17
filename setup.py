import os
import gdown
import shutil

# down load data, create new directory called data

# down load data, create new directory called data

url = 'https://drive.google.com/uc?id=10sxUzJ3BgAFUKx9lYMuTXTJdq8yGYkvy'
output = 'data_download.zip'

gdown.download(url, output, quiet=False)

os.system('unzip data_download')

# make data folder
os.system('mkdir data')

# move unzipped data to data folder
# for windows
# shutil.move(r'Annotations_sub', r'data')
# shutil.move(r'Main', r'data')
# shutil.move(r'JPEGImages_sub', r'data')

# for linux
os.system(r'mv Annotations_sub data')
os.system(r'mv Main data')
os.system(r'mv JPEGImages_sub data')



# move xml script inside and run it to create 

#shutil.copy("datahelper/xmltoyolo.py","data/")

#os.system("python3 data/xmltoyolo.py")

# split data into test and train accordingly

os.mkdir("data/train")
os.mkdir("data/train/images")
os.mkdir("data/train/labels")


# move all training sub images to one folder

with open("data/main/train1.txt") as f:
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

with open("data/main/train1.txt") as f:
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

with open("data/main/test1.txt") as f:
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

with open("data/main/test1.txt") as f:
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

shutil.copy("datahelper/sar_ship_dataset.yaml","yolov7/")
