import numpy as np
import cv2
import os

FOLDER_NAME="swapnil"
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
DIR = os.path.dirname(os.path.abspath(__file__))
train_dir=os.path.join(DIR,"train")
train_dir=os.path.join(train_dir, FOLDER_NAME)

os.chdir(train_dir)

cap = cv2.VideoCapture(0)
count=1
while True:
    count=count+1
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    faces=face_cascade.detectMultiScale(frame)
    for (x,y,w,h) in faces:
        newimg = frame[y:y + h, x:x + w]
        newimg = cv2.resize(newimg, (224, 224))
        newimg=cv2.cvtColor(newimg,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(count)+".jpg", newimg)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("frame", frame)

    ch=cv2.waitKey(1)

    if(ch & 0xFF==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
