import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
#from tensorflow.keras.utils import to_categorical
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout

path = "C:\\Users\\csa54\\OneDrive\\Documents\\CSCI_4366\\dataset\\TrainImageFiles"
images= os.listdir("C:\\Users\\csa54\\OneDrive\\Documents\\CSCI_4366\\dataset\\TrainImageFiles")
emotions = pd.read_csv("C:\\Users\\csa54\\OneDrive\\Documents\\CSCI_4366\\dataset\\trainImages.csv")

#rint(emotions.info())

#split data: 2500 for training and 500 for testing

training = images[:2500]
testing= images[2500:]
trainingcsv = emotions[:2500]
testingcsv = emotions[2500:]
#print(len(training))
#print(training[2499])
#print(trainingcsv.head())
#img = cv2.imread("C:\\Users\\csa54\\OneDrive\\Documents\\CSCI_4366\\dataset\\TrainImageFiles"+"\\"+training[0])
#grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#newimg = grey.reshape(1500, 1980, 1).astype('float32')

x_train = []

#print(newimg.shape)
for i in range(0, 2500):
    img = cv2.imread("C:\\Users\\csa54\\OneDrive\\Documents\\CSCI_4366\\dataset\\TrainImageFiles"+"\\"+training[i])
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(len(grey))
    #newimg = grey.reshape(1500, 1980, 3).astype('float32')
    x_train.append(img)


x_test = []
for i in range(0, 500):
    img = cv2.imread("C:\\Users\\csa54\\OneDrive\\Documents\\CSCI_4366\\dataset\\TrainImageFiles"+"\\"+testing[i])
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x_test.append(img)

#print(img.shape)
#print(img)
#print(training[2499])
#print(testing[0])
#print(testingcsv.head())
#print(trainingcsv['emotion'])
y_train = trainingcsv["emotion"]
y_test = testingcsv["emotion"]

y_train_new = []
#print(y_train[0])
for i in range(0, y_train.size):
    start = np.array([y_train[i]])
    y_train_new.append(start)

y_test_new = []
#print(y_test.head)
for i in range(2501, 3001):
    start = np.array([y_test[i]])
    y_test_new.append(start)
#plt.imshow(grey, cmap='gray')
#print(img.shape)
#print(len(y_test_new))
#plt.show()
print(np.array(x_train).shape)
print(np.array(y_train_new).shape)

model = Sequential()

# Convolutional Phase
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape= (64, 64, 3)))  
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(np.array(x_train), np.array(y_train_new), epochs=10)
score, accurary = model.evaluate(np.array(x_test), np.array(y_test_new))
print(score)
print(accurary)
#model.summary()
