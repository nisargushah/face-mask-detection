import cv2, os
import keras
import numpy as np
import tensorflow as tf
from io import BytesIO
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.mobilenet_v2 import preprocess_input

## We will be using the haarcascade_frontalface_default xml file for own training
## This is because it is extremently efficient
face_cascade = cv2.CascadeClassifier("src/Tensorflow/models/haarcascade_frontalface_default.xml")

#os.chdir("model")
#new_model  = keras.models.clone_model('final.h5')
#str = 'final'
#new_model = load_model(str)
model = keras.models.load_model("model")


## To open up our front camera
capture = cv2.VideoCapture(0)

## Looping to make sure that we continously get feed from the camera

while True:

    # Reading in the data from the front camera
    isTrue, frame = capture.read()
    # The "frame" variable will contain the actual frame that we obtain from the camera
    fps  = capture.get(cv2.CAP_PROP_FPS)

    #model = load_model(new_model)
    # using tbe face_cascade to detect faces
    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.05, minNeighbors=5)
    cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 1, (224, 224, 3), 3, cv2.LINE_AA)
    ##print(fps)
    ##print(len(faces))
    # Faces will be giving us the co-ordinates in which the face is detected.

    ## We are using the for-loopto looparound if we have more than one faces detected
    y = 0
    x = 0
    w = 0
    h = 0
    for _x,_y,_w,_h in faces:

        ## We draw a triangle around the face.
        #img = cv2.rectangle(frame,(_x,_y), (_x+_w,_y+_h), (0,255,0), 3)
        x = _x
        y = _y
        w = _w
        h = _h


    #resized = cv2.resize(img, (int(img.shape[1]/7), int(img.shape[0]/7)))

    # To get the face images cropped out we need to copy it first using .copy()
    if len(faces) > 0:
        clone = frame.copy()

        #now we need to seperate out all the data that is there inside out detected
        # rectangle (anyface that the program detected)
        clone = clone[y:y+h, x:x+w]

        ##print(clone)  #If you see the co-ordinates uncomment the #print statement

        ## now we write the data in a file. You can name the file what ever you want
        ## Don't forget to add the relative path inplace of "1.png" if you want the file
        ## to be stored at a different place
        #clone = cv2.resize(clone, (224,224))
        #print(clone.shape)
        """cv2.imwrite("1.png", clone)
        img = cv2.imread("1.png")"""
        #face = img[startY:endY, startX:endX]
        face = cv2.cvtColor(np.float32(clone), cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        """features = model.predict(face)
        print(features)"""
        (mask, withoutMask) = model.predict(face)[0]


        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        img = cv2.rectangle(frame,(_x,_y), (_x+_w,_y+_h), color, 3)
        cv2.putText(img, label, (_x,_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        # Showing the output
        cv2.imshow("Capture", frame)

        ##If we want to exit the loop (close the video capture) we can press the 'd' key.
        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

## Completing the cv2 code
capture.release()
cv2.destroyAllWindows()
