import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
import numpy as np


my_model=tf.keras.models.load_model('my_model.h5')

img=cv2.imread('open.jpg')
img=cv2.resize(img,(224,224))


input_img=np.array(img,dtype=np.uint8).reshape(1,224,224,3)
plt.imshow(img)
plt.show()

input_img=input_img/255.0

prediction=my_model.predict(input_img)                           #testing if the prediction is accurate or not
print(prediction)

