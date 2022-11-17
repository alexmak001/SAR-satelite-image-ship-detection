import os
import shutil 

# down load data, create new directory called data


# move xml script inside and run it to create 

shutil.copy("datahelper/xmltoyolo.py","data/")

os.system("python3 data/xmltoyolo.py")

# split data into test and train accordingly

os.mkdir("data/train")
os.mkdir("data/train/images")
os.mkdir("data/train/labels")


# move all training sub images to one folder

with open("data/Main/train.txt") as f:
    while True:
        line =  f.readline()
        line = line.replace("\n","")
        filename = line + ".jpg"

        original = "data/images" + filename
        
        target = "data/train/images" + filename

        try:
            shutil.move(original, target)
        except:
            print(filename + " NOT FOUND")

        if not line:
            break


# move all training sub annotations to one folder

with open("data/Main/train.txt") as f:
    while True:
        line =  f.readline()
        line = line.replace("\n","")
        filename = line + ".xml"

        original = "data/annotations/" + filename
        
        target = "data/train/labels" + filename

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

with open("data/Main/test.txt") as f:
    while True:
        line =  f.readline()
        line = line.replace("\n","")
        filename = line + ".jpg"

        original = "data/images" + filename
        
        target = "data/test/images" + filename

        try:
            shutil.move(original, target)
        except:
            print(filename + " NOT FOUND")

        if not line:
            break


# move all test sub annotations to one folder

with open("data/Main/test.txt") as f:
    while True:
        line =  f.readline()
        line = line.replace("\n","")
        filename = line + ".xml"

        original = "data/annotations/" + filename
        
        target = "data/test/labels" + filename

        try:
            shutil.move(original, target)
        except:
            print(filename + " NOT FOUND")

        if not line:
            break
        

# download yolov7, copy data and yaml, edit utils/loss.py

os.system("git clone https://github.com/WongKinYiu/yolov7")

os.system("rm yolov7/utils/loss.py")
shutil.move("datahelper/loss.py", "yolov7/utils/")

shutil.copy("data/train","yolov7/")
shutil.copy("data/test","yolov7/")