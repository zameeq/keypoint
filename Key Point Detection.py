#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential,Model
from keras.layers import Activation,Convolution2D,MaxPooling2D,BatchNormalization,Flatten,Dense,Dropout,Conv2D,MaxPool2D,ZeroPadding2D
from keras.layers import GlobalAveragePooling2D
from tqdm import tqdm

train_data='E:/image/head/'
test_data='E:/image/marple'

def train_data_with_label():
    train_images =[]
    for i in tqdm(os.listdir(train_data)):
        path=os.path.join(train_data,i)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        train_images.append([np.array(img)])
    return train_images
def test_data_with_label():
    test_images =[]
    for i in tqdm(os.listdir(test_data)):
        path=os.path.join(test_data,i)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        test_images.append([np.array(img)])
    return test_images
training_images = train_data_with_label()
test_images=test_data_with_label()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,64,64,1)
tr_lbl_data = np.array([i[0] for i in training_images])

tst_img_data = np.array([i[0] for i in test_images]).reshape(64,64,1)
tst_lbl_data = np.array([i[1] for i in test_images])

face_cascade=cv2.CascadeClassifier('C:/Users/user/Downloads/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
bounding_boxes = face_cascade.detectMultiScale(img_array, 1.25, 6)

from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential,Model
from keras.layers import Activation,Convolution2D,MaxPooling2D,BatchNormalization,Flatten,Dense,Dropout,Conv2D,MaxPool2D,ZeroPadding2D,AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split as tts
#from tensorflow import keras,layers,GlobalAveragePooling2D


model=Sequential()

model.add(BatchNormalization(input_shape=(96,96,1)))
model.add(Convolution2D(24,5,5,border_mode="same",
                       init='he_normal',input_shape=(96,96,1),dim_ordering="tf"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode="valid"))



model.add(Convolution2D(36,5,5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode="valid"))

model.add(Convolution2D(48,5,5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode="valid"))



model.add(Convolution2D(64,3,3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode="valid"))


model.add(Convolution2D(64,3,3))
model.add(Activation("relu"))

model.add(GlobalAveragePooling2D())

model.add(Dense(500,activation="relu"))
model.add(Dense(90,activation="relu"))
model.add(Dense(30))


model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])

checkpointer=ModelCheckpoint(filepath='face_model.h5',verbose=1,save_best_only=True)

epochs=30

hist = model.fit(x=tr_img_data,y=tr_lbl_data, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
    



'''
img_size=500
new_size=cv2.resize(img_array,(img_size,img_size))
plt.imshow(new_size)
plt.show()
'''


# In[ ]:





# In[7]:





# In[ ]:




