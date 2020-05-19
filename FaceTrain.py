import numpy as np
import cv2
import os

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
DIR = os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(DIR,"images")
train_dir=os.path.join(DIR,"train")

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

for root,dirs,files in os.walk(image_dir):
    for file in files:
        path=os.path.join(root,file)
        label=os.path.basename(root)
        #print(label,path)
        img=cv2.imread(path,0)
        new_dir=os.path.join(train_dir,label)
        #print(new_dir)
        faces=face_cascade.detectMultiScale(img)
        os.chdir(new_dir)
        for (x,y,w,h) in faces:
            #print(x,y,w,h)
            newimg=img[y:y+h,x:x+w]
            newimg=cv2.resize(newimg,(224,224))
            cv2.imwrite(file,newimg)

cv2.waitKey(0)
cv2.destroyAllWindows()