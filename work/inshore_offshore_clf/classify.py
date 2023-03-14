import joblib
import numpy as np

def inshore_offshore_classifier(img):
    """
    Takes in an image and classifies it as either offshore(1) or inshore(0)
    """
    # load in predictor
    clf = joblib.load("inshore_offshore_clf_normal_model.pkl")
    
    img_vals = np.copy(img)
    img_50 = np.percentile(img_vals,50)
    img_80 = np.percentile(img_vals,80)
    img_90 = np.percentile(img_vals,90)
    img_30 = np.percentile(img_vals,30)
    
    features = np.array([[img_50, img_80, img_90, img_30]])
    return clf.predict(features)