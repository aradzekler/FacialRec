import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D
import os

NUM_CLASSES = 7 # our Y -> Angry, Sad, Happy..
IMG_ROWS, IMG_COLS = 48, 48  # images are 48x48
BATCH_SIZE = 32  # number of samples to feed in an epoch
TRAIN_DIR = 'C:/Users/97254/PyCharmProjects/FacialRec/images/train'
VALIDATION_DIR = 'C:/Users/97254/PyCharmProjects/FacialRec/images/validation'

# going to add more images to our dataset by manipulating existing ones
train_data_generator = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')


validation_data_generator = ImageDataGenerator(rescale=1./255)

# Downloading the data from the folders.
train_generator = train_data_generator.flow_from_directory(
					TRAIN_DIR,
					color_mode='grayscale',
					target_size=(IMG_COLS, IMG_ROWS),
					batch_size=BATCH_SIZE,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_data_generator.flow_from_directory(
							VALIDATION_DIR,
							color_mode='grayscale',
							target_size=(IMG_COLS, IMG_ROWS),
							batch_size=BATCH_SIZE,
							class_mode='categorical',
							shuffle=True)


# Using Keras Sequential API for building the model
model = Sequential()