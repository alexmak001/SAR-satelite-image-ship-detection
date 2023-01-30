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
