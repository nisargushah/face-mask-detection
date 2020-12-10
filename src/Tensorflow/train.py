"""

@author: Nisarg Shah

Purpose: The main aim of this code is to train out neural network to get the
          maximum possible accuracy along with low compute times


Goal: Stable and accuracte model

"""

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
train_data = []
train_label = []
test_label = []
test_data = []
print(os.listdir(with_mask_path))
for file in os.listdir(with_mask_path):
    print(os.path.join(with_mask_path, file))
    img = load_img(os.path.join(with_mask_path, file), target_size=(224,224))
    img = img_to_array(img)
    #print(file)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)

    train_data.append(img)
    train_label.append(0)

for file in os.listdir(without_mask_path):
    img = load_img(os.path.join(without_mask_path, file), target_size=(224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    #print(file)

    train_data.append(img)
    train_label.append(1)

for file in os.listdir(test_path):
    img = load_img(os.path.join(test_path, file), target_size=(224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    #print(file)

    test_data.append(img)
    test_label.append(0)


for file in os.listdir(test_path_w):
    img = load_img(os.path.join(test_path_w, file), target_size=(224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    print(file)
    test_data.append(img)
    test_label.append(1)


train_data = np.asarray(train_data, dtype=np.float32)
train_label = np.asarray(train_label)
test_data = np.asarray(test_data, dtype=np.float32)
test_label = np.asarray(test_label)
#train_data,train_label, test_data, test_label = shuffle(train_data,train_label, test_data, test_label, random_state=42)
print(len(np.unique(test_label)))
binary = LabelBinarizer()

train_label = binary.fit_transform(train_label)
train_label = to_categorical(train_label)

test_label = binary.fit_transform(test_label)
test_label = to_categorical(test_label)


datagen  = ImageDataGenerator()
base = MobileNetV2(weights="imagenet", include_top = False, input_tensor = Input(shape=(224,224,3)))


head = base.output

head = AveragePooling2D(pool_size=(7,7))(head)
head = Flatten(name="flatten")(head)
head = Dense(64, activation='relu')(head)
head = Dropout(0.7)(head)
head = Dense(2, activation='softmax')(head)

model = Model(inputs=base.input, outputs=head)

for layer in base.layers:
    layer.trainable = False

opt = Adam(lr=1e-4, decay=1e-4/5)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])  #https://stackoverflow.com/questions/61742556/valueerror-shapes-none-1-and-none-2-are-incompatible


out = model.fit(datagen.flow(train_data, train_label,batch_size=32),steps_per_epoch =len(train_label)// 32,epochs=5, validation_data=datagen.flow(test_data, test_label,batch_size=32), validation_steps = len(test_label)// 32)


model.save('../../model')

#ans = model.predict("../../1.png")

print(len(test_data))

"""

Referecnes


https://keras.io/api/preprocessing/image/


"""
