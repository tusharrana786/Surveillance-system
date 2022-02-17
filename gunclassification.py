#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing the required libraries.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection 
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D,BatchNormalization,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as tts
from tensorflow.keras.optimizers import Adam
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import os
import cv2


# In[ ]:


#define a function to load the images from the system in a list.
def load_images_from_folder(folder):
    images = []
    for filenam in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filenam))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (416,416) )
        if img is not None:
            images.append(img)

    return images


# In[ ]:


#loading images.
img1 = load_images_from_folder(r'C:\Users\Pranav\OneDrive\Desktop\New folder\gun-data\0')
img2 = load_images_from_folder(r'C:\Users\Pranav\OneDrive\Desktop\New folder\gun-data\1')
#creating output labels. 0 = not gun, 1 = gun
Y1 = [0]*3000
Y2 = [1]*3000
Y = Y1+Y2
img = img1+img2


# In[ ]:


#convert list to numpy array.
img = np.array(img)
Y = np.array(Y)


# In[ ]:


#converting outputs to binary and using one hot encoding.
Y = LabelBinarizer().fit_transform(Y)
Y = OneHotEncoder().fit_transform(Y).toarray()


# In[ ]:


#define a model to draw rectangle around the (detected) gun.
basemodel = tf.keras.applications.ResNet50V2(include_top = False, input_shape = (416,416,3),weights = 'imagenet')
#uses the predifened weights and freezes them
basemodel.trainable = False
#Decreses the depth of the neural network to one.
x = Flatten()(basemodel.output)
#output the co-ordinates,height and width of the rectangle.
x = Dense(4)(x)
model2 = tf.keras.Model(basemodel.input, x)
#compile the model by defining the loss function and learning rate.
model2.compile(loss='MeanSquaredError', optimizer=Adam(lr=0.00001))
#trains the model.
model2.fit(images,Y,validation_split = 0.2, epochs =5)


# In[ ]:


#loading and preprocessing an image for testing.
img1 = cv2.imread('copy-space-roses-flowers_23-2148860032.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1, (416,416) )


# In[ ]:


#predicting. returns 2D array.
#prob(0) = y_hat[0][0]
#prob(1) = y_hat[0][1]
y_hat = model2.predict(img1.reshape(-1,416,416,3))
print(y_hat)


# In[ ]:


#save.
model2.save('gun_detection.h5')

