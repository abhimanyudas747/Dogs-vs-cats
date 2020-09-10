# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:41:46 2020

@author: Abhimanyu
"""

import cv2
import joblib
import numpy as np

classifier = joblib.load('dogs-vs-cat vgg16-9237.pkl')

def predict(url):
    img = io.imread(url)
    img = cv2.resize(img, (256,256))
    img = img/255
    plt.imshow(img)
    plt.show()
    pred = classifier.predict(img.reshape(1,256,256,3))
    if(np.argmax(pred) == 0):
        print("Prediction : Cat","Confidence: {}%".format(round(pred[0][0]*100,2)))
    else:
        print("Prediction : Dog", "Confidence: {}%".format(round(pred[0][1]*100,2)))
