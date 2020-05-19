# Real-time-Face-Recognition
Real time Face Recognition using OpenCV and Keras

<img src="rd_img/opencv1.png" width='45%'> <img src="rd_img/opencv2.png" width='45%'>

## Getting Started
Clone or download the project to your system

<h4>Prequisites</h4>

<p>Python</p><pre>Install Python 3 or Anaconda</pre>

<h4>Installing</h4>

<p>Install Dependencies</p><pre>pip install -r requirements.txt</pre>
 
<h4>File Strucure</h4>
<img src="rd_img/hierarchy.png" width='50%'>
<p>Create Folders images and train as shown in File Structure</p>

<h4>Running</h4>
<p>FaceTrain.py</p><pre>Checks all the files in the subfolders of images folder and saves the detected faces in train folder. This faces are saved in respective label subfolders in train folder and is used to train and test the model</pre>
<p>Custom.py</p><pre>Captures training images in real time and stores in FOLDER_NAME, Change FOLDER_NAME to label</pre>
<p>ModelTrain.py</p><pre>Trains the model using Keras VGG16</pre>
<p>FaceRecognition.py</p><pre>A frontend video frame which recognizes faces in real time</pre>
