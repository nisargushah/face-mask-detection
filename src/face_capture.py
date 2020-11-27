"""

@author: Nisarg Shah

Purpose: This file will be sued to capture the face of an individual using their
        device's camera using opencv

Goal:  The idea behind this is that we can extract the faces from live feed
        using the camera and then feed it into our deep learning model
        that aslo has face images only and train it for better accuracy

Date: 27th November 2020

"""


import cv2

## We will be using the haarcascade_frontalface_default xml file for own training
## This is because it is extremently efficient
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

## To open up our front camera
capture = cv2.VideoCapture(0)

## Looping to make sure that we continously get feed from the camera

while True:

    # Reading in the data from the front camera
    isTrue, frame = capture.read()
    # The "frame" variable will contain the actual frame that we obtain from the camera

    # using tbe face_cascade to detect faces
    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.05, minNeighbors=5)

    # Faces will be giving us the co-ordinates in which the face is detected.

    ## We are using the for-loopto looparound if we have more than one faces detected
    for x,y,w,h in faces:

        ## We draw a triangle around the face.
        img = cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,0), 3)

    #resized = cv2.resize(img, (int(img.shape[1]/7), int(img.shape[0]/7)))

    # To get the face images cropped out we need to copy it first using .copy()
    clone = frame.copy()

    #now we need to seperate out all the data that is there inside out detected
    # rectangle (anyface that the program detected)
    clone = clone[y:y+h, x:x+w]

    #print(clone)  #If you see the co-ordinates uncomment the print statement

    ## now we write the data in a file. You can name the file what ever you want
    ## Don't forget to add the relative path inplace of "1.png" if you want the file
    ## to be stored at a different place
    cv2.imwrite("1.png", clone)
    # Showing the output
    cv2.imshow("Capture", img)

    ##If we want to exit the loop (close the video capture) we can press the 'd' key.
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

## Completing the cv2 code
capture.release()
cv2.destroyAllWindows()
