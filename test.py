from keras.models import load_model
from keras_preprocessing.image import img_to_array
from keras_preprocessing import image
import numpy as np
import cv2  # opencv

import Constants

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
# used to detect our face in a webcam image.
google_face_detector = cv2.CascadeClassifier(r'C:\Users\97254\PyCharmProjects\FacialRec\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\97254\PyCharmProjects\FacialRec\small_vgg_facerec.h5')

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

webcam_capture = cv2.VideoCapture(0)  # 0 for webcam

while True:
	# Grab a single frame of video
	ret, frame = webcam_capture.read()
	labels = []
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = google_face_detector.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in faces: # x,y,w,h -> CNN features
		cv2.rectangle(frame, (x, y), (x + w, y + h), Constants.RED_COLOR, Constants.RECTANGLE_THICKNESS) # drawing a rectangle around the face
		roi_gray = gray[y:y + h, x:x + w] # our face from the webcam is not grey but colored, we color it to grey
		roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA) # size of our face is not 48x48, we need to convert it.
		# rect,face,image = face_detector(frame)

		if np.sum([roi_gray]) != 0: # if we have at least one face
			roi = roi_gray.astype('float') / 255.0 # dividing by 255 gives us a maximum value between 0 and 1
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)

			# make a prediction on the ROI, then lookup the class

			preds = classifier.predict(roi)[0]
			label = class_labels[preds.argmax()]
			label_position = (x, y)
			cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, Constants.RECTANGLE_THICKNESS, Constants.GREEN_COLOR, 3)
		else: # If no face detected.
			cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, Constants.RECTANGLE_THICKNESS, Constants.GREEN_COLOR, 3)
	cv2.imshow('Emotion Detector', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'): # if we press q, the frame will stop!
		break

webcam_capture.release()
cv2.destroyAllWindows()
