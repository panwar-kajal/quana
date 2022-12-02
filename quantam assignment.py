#!/usr/bin/env python
# coding: utf-8

# # QUANTUM ASSIGNMENT

# # QUESTION-1A sorted array is rotated at some unknown point, find the minimum element in it.

# In[2]:


def findMin(arr, N):
     
    min_ele = arr[0];
 
    
    # find minimum element
    for i in range(N) :
        if arr[i] < min_ele :
            min_ele = arr[i]
 
    return min_ele;
 
# Driver program
arr = [5, 6, 1, 2, 3, 4]
N = len(arr)
 
print(findMin(arr,N))
arr= [1, 2, 3, 4]
N = len(arr)
 
print(findMin(arr,N))

arr = [2, 1]
N = len(arr)
 
print(findMin(arr,N))


# # Q2.Given two strings str1 and str2 and below operations that can performed on str1. Find minimum number of edits (operations) required to convert ‘str1′ into ‘str2

# In[6]:


def minCost(str1, str2, n):
 
    cost = 0
 
    # For every character of str1
    for i in range(n):
 
        # If current character is not
        # equal in both the strings
        if (str1[i] != str2[i]):
 
            # If the next character is also different in both
            # the strings then these characters can be swapped
            if (i < n - 1 and str1[i + 1] != str2[i + 1]):
                swap(str1[i], str1[i + 1])
                cost += 1
             
            # Change the current character
            else:
                cost += 1
             
    return cost
 
# Driver code
if __name__ == '__main__':
 
    str1 = "Quantom"
    str2= "Quantum"
    
    n = len(str1)
 
    print(minCost(str1, str2, n))
    
    


# # Predict the below image with Machine Learning Algorithm and Check can you apply PCA?i) check the accuracy.ii) if accuracy level is low then how can we increase the accuracy 

# In[2]:


get_ipython().system(' pip install keras')


# In[8]:


get_ipython().system('pip install tensorflow')


# In[53]:


get_ipython().system('pip install pydot')


# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense,Flatten
#from keras.model import Sequential
#from keras_utils import to_categorical
from keras.datasets import mnist
from keras import utils
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical


# In[41]:


(X_train,y_train), (X_test ,y_test) = mnist.load_data()


# In[42]:


X_test.shape


# In[43]:


fig, axes=plt.subplots(ncols=10,sharex=False, 
     sharey=True, figsize=(20,4))
for i in range(10):
    axes[i].set_title(y_train[i])
    axes[i].imshow(X_train[i], cmap='Blues')
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
plt.show()


# In[44]:


y_train= to_categorical(y_train)
y_test = to_categorical(y_test)


# In[45]:


y_test.shape


# In[46]:


model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(10 , activation='sigmoid'))
model.add(Dense(5 ,activation='sigmoid'))
model.add(Dense(10 , activation='softmax'))


# In[48]:


model.summary()


# In[60]:


model.compile(loss='categorical_crossentropy',
                    optimizer='adam' ,
                    metrics=['acc']    )


# In[67]:


history=model.fit(X_train,y_train , epochs=20,
         validation_data=(X_test,y_test) )


# In[68]:


model.save('mnist_model.h5')


# In[69]:


plt.plot(history.history['loss'],label='train Loss')
plt.plot(history.history['val_loss'],label='validation Loss')
plt.legend()


# In[70]:


plt.plot(history.history['acc'],label='Train acc')
plt.plot(history.history['val_acc'],label='validation acc')
plt.legend()


# In[114]:


plt.imshow(X_test[18],cmap='Blues')


# In[115]:


X_test[18].shape


# In[117]:


x=np.reshape(X_test[18], (1,28,28))
np.argmax(model.predict(x))


# In[118]:


model.predict(x)


# # Q5. open the first link which is detect lane with live camera and in second link video you have to recognize lane detect with simple line.
# 

# In[8]:


get_ipython().system('pip install opencv-python')


# In[49]:


import matplotlib.pylab as plt
import cv2
import numpy as np


# In[50]:


image = cv2.imread('road.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [
    (0, height),
    (width/2, height/2),
    (width, height)
]


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

cropped_image = region_of_interest(image,
                np.array([region_of_interest_vertices], np.int32),)

plt.imshow(cropped_image)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Handwritten Digit Recognition by MNIST")

# Decide if to load an existing model or to train a new one
train_new_model = True

if train_new_model:
    # Loading the MNIST data set with samples and splitting it
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizing the data (making length = 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Create a neural network model
    # Add one flattened input layer for the pixels
    # Add two dense hidden layers
    # Add one dense output layer for the 10 digits
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compiling and optimizing model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=3)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Saving the model
    model.save('handwritten_digits.model')
else:
    # Load the model
    model = tf.keras.models.load_model('handwritten_digits.model')

# Load custom images and predict them
image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1


# In[ ]:




