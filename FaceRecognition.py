import numpy as np
import cv2
from keras.models import load_model
import pickle
from keras.preprocessing.image import img_to_array

mlb = pickle.loads(open("mlb.pickle", "rb").read())
model=load_model('face_model.h5')
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    faces=face_cascade.detectMultiScale(frame)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        new_img=frame[y:y+h,x:x+w]
        new_img=cv2.resize(new_img,(224,224))
        new_img=new_img.astype("float") / 255.0
        new_img = img_to_array(new_img)
        new_img = np.expand_dims(new_img, axis=0)

        proba = model.predict(new_img)[0]
        idxs = np.argsort(proba)[::-1][:2]
        for (i, j) in enumerate(idxs):
            # build the label and draw the label on the image
            label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
            if(proba[j] * 100>70):
                cv2.putText(frame, label, (10, (i * 30) + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    cv2.imshow("frame", frame)

    ch=cv2.waitKey(1)

    if(ch & 0xFF==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
