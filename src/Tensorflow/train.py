"""

@author: Nisarg Shah

Purpose: The main aim of this code is to train out neural network to get the
          maximum possible accuracy along with low compute times


Goal: Stable and accuracte model

"""
## Step 1: Loading out libraies
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D, Dense, Dropout,Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.utils import shuffle


### Reading in all the images
with_mask_path = '../../dataset/with_mask'
without_mask_path = '../../dataset/without_mask'
test_path = '../../dataset/test/with_mask'
test_path_w = '../../dataset/test/without_mask'

## Declaring our lists to store the data
train_data = []
train_label = []
test_label = []
test_data = []


#print(os.listdir(with_mask_path))  ## Quick debug check to see if the code behaves properly

"""

For the next 4 for loops, we have decided to take in the data from their desired
folders and take in the data from there and read the images, convert them to
arrays and then pass them through the preprocess_inputs to make sure that it is
in proper format for our model to train on it later


"""

## os.listdir(folder) list all the files that are there in that folder, this makes
## it easier for us to load all the files. This option can also be used - os.walk(folder)
## You can fimd more on  that via this link - https://www.tutorialspoint.com/python3/os_walk.htm
for file in os.listdir(with_mask_path):

    ## Quick print statement for debugging
    #print(os.path.join(with_mask_path, file))


    ## We will be using the keras load_img to load the images.
    ## Here please note that I am using os.path.join but it is not mandatory
    ## We can edit this and use load_img(file) but sometiems while training on
    ## cloud this doesn't work. So the extra work

    img = load_img(os.path.join(with_mask_path, file), target_size=(224,224)) ## Our dataset is 224x244 frame
    img = img_to_array(img) ##Converting the image to array for our calculations

    #print(file)

    ## The next step is optional since we are not really concerneed with detecting the color of the face mask.
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ##  Sending it to keras, to preprocess
    img = preprocess_input(img)

    ## Appending it to our list
    train_data.append(img)
    train_label.append(0)  # We are appending 0 for mask and 1


"""
THis follows the same comments as above for loop.

"""
for file in os.listdir(without_mask_path):
    img = load_img(os.path.join(without_mask_path, file), target_size=(224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    #print(file)

    train_data.append(img)
    train_label.append(1)

"""
THis follows the same comments as above for loop.

"""
for file in os.listdir(test_path):
    img = load_img(os.path.join(test_path, file), target_size=(224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    #print(file)

    test_data.append(img)
    test_label.append(0)


"""
THis follows the same comments as above for loop.

"""
for file in os.listdir(test_path_w):
    img = load_img(os.path.join(test_path_w, file), target_size=(224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    print(file)
    test_data.append(img)
    test_label.append(1)

### Converting our list to arrays as float32. If speed is a major concern, we can
### switch to float16

train_data = np.asarray(train_data, dtype=np.float32)
train_label = np.asarray(train_label)
test_data = np.asarray(test_data, dtype=np.float32)
test_label = np.asarray(test_label)
#train_data,train_label, test_data, test_label = shuffle(train_data,train_label, test_data, test_label, random_state=42)


# print(len(np.unique(test_label))) ##Quick debug check

## We need to convert the labels to categoies. LabelBinarizer is the easiest ways
## but we can also use One hot Encoder for this
binary = LabelBinarizer()

## We fit our label list to binary and then convert it to categorical using keras
train_label = binary.fit_transform(train_label)
train_label = to_categorical(train_label)

test_label = binary.fit_transform(test_label)
test_label = to_categorical(test_label)


## We use ImageGenerator to send our data to out model. We will leave everything
## on default but we can change them. Please visit documentation to fit your use case
datagen  = ImageDataGenerator()

## We will be using MobileNetV2 and for the weights we will be using the standard
## imagenet weights and we won;t be training any of the layers
base = MobileNetV2(weights="imagenet", include_top = False, input_tensor = Input(shape=(224,224,3)))

## We would train some layers however to fit out use case and mainly for the
## Final Dense layer to get 2 classes
base_output = base.output

### Standard layers used. Please note that we will be underfitting the data in this.
### You can tune the parameters to fit our model better
base_output = AveragePooling2D(pool_size=(7,7))(base_output)
base_output = Flatten()(head)
base_output = Dense(64, activation='relu')(base_output)
base_output = Dropout(0.7)(base_output)
base_output = Dense(2, activation='softmax')(base_output)

model = Model(inputs=base.input, outputs=base_output)

### We want all the layers to be not trainable.
for layer in base.layers:
    layer.trainable = False


## We will be using Adam as an optimizer. Adam is always reliable but SGD can also
## be used in this.
opt = Adam(lr=1e-4, decay=1e-4/5)

## Compiling the model
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])  #https://stackoverflow.com/questions/61742556/valueerror-shapes-none-1-and-none-2-are-incompatible

## Fitting our model
out = model.fit(datagen.flow(train_data, train_label,batch_size=32),steps_per_epoch =len(train_label)// 32,epochs=5, validation_data=datagen.flow(test_data, test_label,batch_size=32), validation_steps = len(test_label)// 32)

## Saving our model
model.save('../../model')

#ans = model.predict("../../1.png")

#print(len(test_data))

"""

References


https://keras.io/api/preprocessing/image/


"""
