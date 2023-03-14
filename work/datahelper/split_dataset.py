import random
import glob
import os
import shutil


def copyfiles(fil, root_dir):
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]

    # copy image
    src = fil
    dest = os.path.join(root_dir, image_dir, f"{filename}.jpg")
    shutil.copyfile(src, dest)

    # copy annotations
    src = os.path.join(label_dir, f"{filename}.txt")
    dest = os.path.join(root_dir, label_dir, f"{filename}.txt")
    if os.path.exists(src):
        shutil.copyfile(src, dest)


label_dir = "Annotations_sub_yolo/"
image_dir = "JPEGImages_sub/"
lower_limit = 0
files = glob.glob(os.path.join(image_dir, '*.jpg'))

random.shuffle(files)

folders = {"train": 0.7, "val": 0.2, "test": 0.1}
check_sum = sum([folders[x] for x in folders])
check_sum = round(check_sum,4)
assert check_sum == 1.0, "Split proportion is not equal to 1.0"

for folder in folders:
    os.mkdir(folder)
    temp_label_dir = os.path.join(folder, label_dir)
    os.mkdir(temp_label_dir)
    temp_image_dir = os.path.join(folder, image_dir)
    os.mkdir(temp_image_dir)

    limit = round(len(files) * folders[folder])
    for fil in files[lower_limit:lower_limit + limit]:
        copyfiles(fil, folder)
    lower_limit = lower_limit + limit

os.rename("train/Annotations_sub_yolo","train/labels")
os.rename("train/JPEGImages_sub","train/images")

os.rename("val/Annotations_sub_yolo","val/labels")
os.rename("val/JPEGImages_sub","val/images")

os.rename("test/Annotations_sub_yolo","test/labels")
os.rename("test/JPEGImages_sub","test/images")

# os.mkdir("images")
# os.mkdir("labels")

# shutil.move("train/Annotations_sub_yolo/", "labels/")
# os.rename("labels/Annotations_sub_yolo/","labels/train/")

# shutil.move("train/JPEGImages_sub", "images/")
# os.rename("images/JPEGImages_sub","train")

# shutil.move("test/Annotations_sub_yolo", "labels/")
# os.rename("labels/Annotations_sub_yolo","test")

# shutil.move("test/JPEGImages_sub", "images/")
# os.rename("images/JPEGImages_sub","test")

# shutil.move("val/Annotations_sub_yolo", "labels/")
# os.rename("labels/Annotations_sub_yolo","val")

# shutil.move("val/JPEGImages_sub", "images/")
# os.rename("images/JPEGImages_sub","val")