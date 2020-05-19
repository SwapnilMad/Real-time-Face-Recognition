from keras.layers import Dense,Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = [224,224]

data=[]
labels=[]

DIR = os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(DIR,"train")

for root,dirs,files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root, file)
        label = os.path.basename(root)
        img=cv2.imread(path)
        img = img_to_array(img)
        data.append(img)
        labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

mlb = LabelBinarizer()
labels = mlb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)


vgg = VGG16(input_shape=IMAGE_SIZE+[3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable=False

x = Flatten()(vgg.output)
prediction = Dense(labels.shape[1],activation='softmax')(x)

model=Model(input=vgg.input,outputs=prediction)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',metrics=['accuracy'])


datagen=ImageDataGenerator(shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)


r=model.fit_generator(datagen.flow(trainX, trainY,batch_size = 32),
                      validation_data=(testX, testY),
                      epochs=5,
                      steps_per_epoch=len(trainX),
                      validation_steps=len(testX))


model.save('face_model.h5')

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
#plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
#plt.savefig('AccVal_acc')

f = open("mlb.pickle", "wb")
f.write(pickle.dumps(mlb))
f.close()