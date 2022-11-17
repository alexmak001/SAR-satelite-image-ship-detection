import os
import gdown
import shutil

# down load data, create new directory called data

url = 'https://drive.google.com/uc?id=10sxUzJ3BgAFUKx9lYMuTXTJdq8yGYkvy'
output = 'data_download.zip'

gdown.download(url, output, quiet=False)

os.system('unzip data_download')

# make data folder
os.system('mkdir data')

# move unzipped data to data folder
# for windows
shutil.move(r'Annotations_sub', r'data')
shutil.move(r'Main', r'data')
shutil.move(r'JPEGImages_sub', r'data')

# for linux
os.system(r'mv Annotations_sub data')
os.system(r'mv Main data')
os.system(r'mv JPEGImages_sub data')