import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import joblib

from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans
# from sklearn import svm
# from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn import metrics
# from sklearn.preprocessing import binarize

# given a file path of jpg images, load in imgs, then predict on them.

def main(targets):
    # targets == filepath to the downloaded images

    clf = joblib.load("inshore_offshore_clf_model.pkl")

    fp = targets

    # read in images
    fnames = os.listdir(fp)
    train = []
    for file in fnames:
        train.append(plt.imread(fp + '/{}'.format(file)))
    
    img_df = pd.DataFrame(columns = ['50th', '80th', '90th', '30th'])
    
    for img in train:
        img_vals = np.copy(img)
        img_50 = np.percentile(img_vals,50)
        img_80 = np.percentile(img_vals,80)
        img_90 = np.percentile(img_vals,90)
        img_30 = np.percentile(img_vals,30)

        img_df.append({'50th': img_50, '80th': img_80, '90th': img_90, '30th': img_30}, ignore_index=True)

    return clf.predict(img_df)


if __name__ == '__main__':
    # run via:
    # python run.py
    targets = sys.argv[1]
    main(targets)