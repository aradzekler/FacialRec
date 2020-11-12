import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

NUM_CLASSES = 7  # our Y -> Angry, Sad, Happy..
IMG_ROWS, IMG_COLS = 48, 48  # images are 48x48
BATCH_SIZE = 16  # number of samples to feed in an epoch
TRAIN_DIR = 'C:/Users/97254/PyCharmProjects/FacialRec/images/train'
VALIDATION_DIR = 'C:/Users/97254/PyCharmProjects/FacialRec/images/validation'
EPOCHS = 50
NUM_TRAIN_SAMPLES = 28821
NUM_VALIDATION_SAMPLES = 7066

# going to add more images to our dataset by manipulating existing ones
train_data_generator = ImageDataGenerator(
	rescale=1. / 255,  # maximum channels: 255
	rotation_range=30,
	shear_range=0.3,  # like tilting the image
	zoom_range=0.3,
	width_shift_range=0.4,  # off-centering the image
	height_shift_range=0.4,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode='nearest',
	validation_split=0.2)

# we dont need to fake images for validation..
test_data_generator = ImageDataGenerator(rescale=1. / 255)

# Downloading the data from the folders and giving the model additional details on our data.
train_generator = train_data_generator.flow_from_directory(
	TRAIN_DIR,
	color_mode='grayscale',
	target_size=(IMG_ROWS, IMG_COLS),
	batch_size=BATCH_SIZE,
	class_mode='categorical',
	shuffle=True)

validation_generator = train_data_generator.flow_from_directory(
	VALIDATION_DIR,
	color_mode='grayscale',
	target_size=(IMG_ROWS, IMG_COLS),
	batch_size=BATCH_SIZE,
	class_mode='categorical',
	subset='validation')

# Using Keras Sequential API for building the model
'''
Max Pooling - Downsizing our image to train the model better,
	ease on the memory and give less features.
Batch Normalization - Normalizes our batch distribution (so we will not get 
	a batch full of angry faces..)
ELU (Exponential Linear Unit) - Our Activation function - 
	'Smoother' Leaky ReLu with values below zero -> a*(e^x - 1)
Softmax - Activation for outputting a categorical output - by using
	a range between 0 to 1 giving the categories a probability 
	(80% angry, 19% sad, 1% happy for example)
Dropout - dropping some precent of neurons to make training harder
	and better for our training.
Padding - Sometimes filter does not fit perfectly fit the input image,
	to solve this we pad the image with zeros so the filter will fit. 
Filter - Matrix/Field of numbers (weights). If we think of the filter as 
	a flashlight that we point to the image, the place it lights up is 
	called a 'Receptive Field'. we move on the image with our filter
	or - 'convolving', and creating an output field to use as an input
	to the next layers. here the filter size is 3x3 pixels.
Stride - Over how much pixels our filter shifts when he is convolving
	the image.

'''
model = Sequential()

# Block-1

model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(IMG_ROWS, IMG_COLS, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(IMG_ROWS, IMG_COLS, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block-2

model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block-4

model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block-5

model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6

model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

model.add(Dense(NUM_CLASSES, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

# will create a file checkpoint for our model, it will overwrite it every run until we will find the best model
checkpoint = ModelCheckpoint('small_vgg_facerec.h5',
                             monitor='val_loss', # monitor our progress by loss value.
                             mode='min', # smaller loss is better, we try to minimize it.
                             save_best_only=True,
                             verbose=1)

# if our model accuracy (loss) is not improving over 3 epochs, stop the training, something is fishy
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

# if our loss is not improving, try to reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [checkpoint, earlystop, reduce_lr]

print("\nLoading model...")
model.compile(loss='categorical_crossentropy',
              optimizer = SGD(lr=0.001, momentum=0.9),
              metrics=['accuracy'])

history = model.fit_generator(
                train_generator,
                steps_per_epoch=NUM_TRAIN_SAMPLES//BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=NUM_VALIDATION_SAMPLES//BATCH_SIZE)