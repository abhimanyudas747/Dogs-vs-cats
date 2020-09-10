# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:41:46 2020

@author: Abhimanyu
"""

import cv2
import joblib
import numpy as np
img = io.imread('https://cdn.kinsights.com/cache/93/9a/939a00a381fe6ec68af0a319bc4e3a15.jpg')
img = cv2.resize(img, (256,256))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



car_classifier = cv2.CascadeClassifier('face.xml')
found = car_classifier.detectMultiScale(img_gray) 

for (x,y,width,height) in found:
    cv2.rectangle(img_rgb, (x, y),  
                      (x + height, y + width),  
                      (255, 0, 0), 2) 


img_rgb = img_rgb / 255

classifier = joblib.load('in-use9257.pkl')

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
