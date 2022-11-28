import os
import shutil
# move test images to offshore

os.mkdir('data/test_offshore')
os.mkdir('data/test_inshore')
os.mkdir('data/test_offshore/images')
os.mkdir('data/test_inshore/images')
os.mkdir('data/test_offshore/labels')
os.mkdir('data/test_inshore/labels')
with open("data/main/test_offshore.txt") as f:
    while True:
        line =  f.readline()
        line = line.replace("\n","")
        filename = line + ".jpg"

        original = "data/test/images/" + filename
        
        target = "data/test_offshore/images/" + filename

        try:
            shutil.move(original, target)
        except:
            print(filename + " NOT FOUND - offshore")

        if not line:
            break


# move all test sub annotations to one folder

with open("data/main/test_offshore.txt") as f:
    while True:
        line =  f.readline()
        line = line.replace("\n","")
        filename = line + ".txt"

        original = "data/test/labels/" + filename
        
        target = "data/test_offshore/labels/" + filename

        try:
            shutil.move(original, target)
        except:
            print(filename + " NOT FOUND - offshore")

        if not line:
            break

with open("data/main/test_inshore.txt") as f:
    while True:
        line =  f.readline()
        line = line.replace("\n","")
        filename = line + ".jpg"

        original = "data/test/images/" + filename
        
        target = "data/test_inshore/images/" + filename

        try:
            shutil.move(original, target)
        except:
            print(filename + " NOT FOUND - offshore")

        if not line:
            break


# move all test sub annotations to one folder

with open("data/main/test_inshore.txt") as f:
    while True:
        line =  f.readline()
        line = line.replace("\n","")
        filename = line + ".txt"

        original = "data/test/labels/" + filename
        
        target = "data/test_inshore/labels/" + filename

        try:
            shutil.move(original, target)
        except:
            print(filename + " NOT FOUND - offshore")

        if not line:
            break

os.mkdir('data/train_offshore')
os.mkdir('data/train_inshore')
os.mkdir('data/train_offshore/images')
os.mkdir('data/train_inshore/images')
os.mkdir('data/train_offshore/labels')
os.mkdir('data/train_inshore/labels')
with open("sean_notebooks/train_classified.txt") as f:
    while True:
        line =  f.readline()
        line = line.replace("\n","")
        line = line.split(',')
        filename = line[0]
        shore = line[1]

        original_img = "data/train/images/" + filename
        original_label = 'data/train/labels/' + filename[:-4] + '.txt'
        if shore == 'offshore':
            target_img = "data/train_offshore/images/" + filename
            target_label = "data/train_offshore/labels/" + filename[:-4] + '.txt'
        else:
            target_img = "data/train_inshore/images/" + filename
            target_label = "data/train_inshore/labels/" + filename[:-4] + '.txt'
        try:
            shutil.move(original_img, target_img)
            shutil.move(original_label, target_label)
        except Exception as e:
            print(e)

        if not line:
            break