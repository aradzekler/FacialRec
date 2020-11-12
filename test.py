from keras.models import load_model
from keras_preprocessing.image import img_to_array
from keras_preprocessing import image
import numpy as np
import cv2 # opencv


'''
Object Detection using Haar feature-based cascade classifiers is an effective
object detection method proposed by Paul Viola and Michael Jones in their paper, 
"Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It 
is a machine learning based approach where a cascade function is trained from a 
lot of positive and negative images. It is then used to detect objects in other images.

Here we will work with face detection. Initially, the algorithm needs a lot of positive 
images (images of faces) and negative images (images without faces) to train the classifier. 
Then we need to extract features from it. For this, Haar features shown in the below image are used. 
They are just like our convolutional kernel. Each feature is a single value obtained by subtracting 
sum of pixels under the white rectangle from sum of pixels under the black rectangle.

'''
face_classifier = cv2.CascadeClassifier(r'C:\Users\97254\PyCharmProjects\FacialRec\haarcascade_frontalface_default.xml')
model = load_model(r'C:\Users\97254\PyCharmProjects\FacialRec\small_vgg_facerec.h5')