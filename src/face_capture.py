import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

capture = cv2.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    #cv2.imshow('Video', frame)
    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.05, minNeighbors=5)
    #cv.imshow('Video', frame)
    for x,y,w,h in faces:
        img = cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,0), 3)
        print(x+w,y+w)

    #resized = cv2.resize(img, (int(img.shape[1]/7), int(img.shape[0]/7)))
    clone = frame.copy()
    clone = clone[y:y+h, x:x+w]
    print(clone)
    cv2.imwrite("1.png", clone)
    cv2.imshow("Gray", img)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()
