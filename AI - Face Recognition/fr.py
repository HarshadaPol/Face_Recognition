import cv2 as cv
import os

cascPath=os.path.dirname(cv.__file__)+"/data/haarcascade_frontalface_default.xml"
cap_video = cv.VideoCapture(0)
cascade_face = cv.CascadeClassifier(cascPath)

while True:
    # Capture frame-by-frame
    returnVal, vdoFrames = cap_video.read()

    gray = cv.cvtColor(vdoFrames, cv.COLOR_BGR2GRAY)

    faces = cascade_face.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(vdoFrames, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv.imshow('Video', vdoFrames)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break