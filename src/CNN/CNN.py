#! /usr/bin/python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras.callbacks import *
from scipy.ndimage.filters import *

def fcn8(input_height, input_width, n_classes):
	input_img = Input(shape=(input_height, input_width, 1))
	activation = 'relu'
	
	#Block 1
	x = Conv2D(64, (3,3), activation=activation, padding='same', \
		name='block1_conv1', data_format='channels_last')\
		(input_img)
	x = Conv2D(64, (3,3), activation=activation, padding='same', \
		name='block1_conv2', data_format='channels_last')\
		(x)
	x = MaxPooling2D((2, 2), strides=(2,2), name='block1_pool', \
		data_format='channels_last')(x)

	#Block 2 
	x = Conv2D(128, (3,3), activation=activation, padding='same', \
		name='block2_conv1', data_format='channels_last')\
		(x)
	x = Conv2D(128, (3,3), activation=activation, padding='same', \
		name='block2_conv2', data_format='channels_last')\
		(x)
	x = MaxPooling2D((2, 2), strides=(2,2), name='block2_pool', \
		data_format='channels_last')(x)

	#Block 3
	x = Conv2D(256, (3,3), activation=activation, padding='same', \
		name='block3_conv1', data_format='channels_last')\
		(x)
	x = Conv2D(256, (3,3), activation=activation, padding='same', \
		name='block3_conv2', data_format='channels_last')\
		(x)
	x = Conv2D(256, (3,3), activation=activation, padding='same', \
		name='block3_conv3', data_format='channels_last')\
		(x)
	pool3 = MaxPooling2D((2, 2), strides=(2,2), name='block3_pool', \
		data_format='channels_last')(x)

	#Block 4
	x = Conv2D(512, (3,3), activation=activation, padding='same', \
		name='block4_conv1', data_format='channels_last')\
		(pool3)
	x = Conv2D(512, (3,3), activation=activation, padding='same', \
		name='block4_conv2', data_format='channels_last')\
		(x)
	x = Conv2D(512, (3,3), activation=activation, padding='same', \
		name='block4_conv3', data_format='channels_last')\
		(x)
	pool4 = MaxPooling2D((2, 2), strides=(2,2), name='block4_pool', \
		data_format='channels_last')(x)

	#Block 5
	x = Conv2D(512, (3,3), activation=activation, padding='same', \
		name='block5_conv1', data_format='channels_last')\
		(pool4)
	x = Conv2D(512, (3,3), activation=activation, padding='same', \
		name='block5_conv2', data_format='channels_last')\
		(x)
	x = Conv2D(512, (3,3), activation=activation, padding='same', \
		name='block5_conv3', data_format='channels_last')\
		(x)
	pool5 = MaxPooling2D((2, 2), strides=(2,2), name='block5_pool', \
		data_format='channels_last')(x)

	#Deconvolution pool5
	window_size = int(input_height / 2**5)
	x = (Conv2D(4096, (window_size, window_size), activation=activation, padding='same', \
		name='conv6', data_format='channels_last'))(pool5)

	x = (Conv2D(4096, (1,1), activation=activation, padding='same', \
		name='conv7', data_format='channels_last'))(x)

	pool5_deconv = (Conv2DTranspose(n_classes, kernel_size=(4, 4),\
		strides=(4, 4), use_bias=False, \
		data_format='channels_last'))(x)

	#Deconvolution pool4
	x = (Conv2D(n_classes, (1,1), activation=activation, padding='same', \
		name='pool4_filetered', data_format='channels_last'))(pool4)
	pool4_deconv = (Conv2DTranspose(n_classes, kernel_size=(2, 2),\
		strides=(2, 2), use_bias=False, \
		data_format='channels_last'))(x)

	#Layer Fusion
	pool3_filtered = (Conv2D(n_classes, (1,1), activation=activation, padding='same', \
		name='pool3_filetered', data_format='channels_last'))(pool3)
	x = Add(name='layer_fusion')([pool5_deconv, pool4_deconv, pool3_filtered])

	#8 Times Deconvolution and Softmax
	x = (Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), \
		use_bias=False, data_format='channels_last'))(x)
	x = (Activation('softmax'))(x)

	return Model(input_img, x)

def load_cropped_training():
	x_temp = []
	x_train = []
	y_train = []
	base_frames = '../data/training/cropped/avg_frames/'
	base_masks = '../data/training/cropped/masks/'
	samples = os.listdir(base_frames)
	n_classes = 2
	for sample in samples:	
		x_img = np.load(base_frames + sample).astype(int)
		sample_instance = np.zeros((x_img.shape[0], x_img.shape[1], 1))
		sample_instance[:, :, 0] = x_img
		sample_instance = median_filter(sample_instance, size=3)
		y_img = np.load(base_masks + sample).astype(int)
		mask = np.zeros((y_img.shape[0], y_img.shape[1], n_classes))
		for i in range(n_classes):
			mask[:, :, i] = (y_img == i).astype(int)
		if (np.sum(y_img == 1) / (y_img.shape[0] * y_img.shape[1])) >= 0.4:
			x_train.append(sample_instance)
			y_train.append(mask)
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	return x_train, y_train

def load_cropped_testing():
	x_temp = []
	x_test = []
	region_names = []
	base_frames = '../data/testing/cropped/avg_frames/'
	samples = os.listdir(base_frames)
	n_classes = 2
	for sample in samples:	
		x_img = np.load(base_frames + sample).astype(int)
		sample_instance = np.zeros((x_img.shape[0], x_img.shape[1], 1))
		sample_instance[:, :, 0] = x_img
		sample_instance = median_filter(sample_instance, size=3)
		x_test.append(sample_instance)
		region_names.append(sample[:len(sample) - 4])
	x_test = np.array(x_test)
	return x_test, region_names

def testing_sample_names():
	base = '../data/testing/full/avg_frames/'
	sample_names = os.listdir(base)
	for i in range(len(sample_names)):
		sample_names[i] = sample_names[i][:len(sample_names[i]) - 4]
	return sample_names

def restitch_images(sample_names, region_names, pred_imgs):
	regions = {}
	for i in range(len(region_names)):
		regions[region_names[i]] = pred_imgs[i]
	outputs = []
	for name in sample_names:
		counter = 0
		img = np.zeros((512, 512))
		for i in range(1, 9):
			for j in range(1, 9):
				img[64 * (i - 1): 64 * i, 64 * (j - 1): 64 * j] = \
					regions[name + '-' + str(counter)]
				counter += 1
		outputs.append(img)
	return outputs
	
#Load in Testing and Traning Data, and fit FCN8 model
x_train, y_train = load_cropped_training()
x_test, region_names = load_cropped_testing()
sample_names = testing_sample_names()
model = fcn8(64, 64, 2)
model.summary()
sgd = optimizers.SGD(lr=0.03, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, \
		metrics=['accuracy'])
model_path = '../models/Current_Best.h5'
callbacks=[ModelCheckpoint(filepath=model_path, \
		monitor='val_loss', save_best_only=True)]
model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.1, callbacks=callbacks)
model.save('../models/Full.h5')

#Save results from the model that trained on the full number of epochs
pred = model.predict(x_test)
pred_imgs = np.argmax(pred, axis=3)
outputs = restitch_images(sample_names, region_names, pred_imgs)
for i in range(len(outputs)):
	cv2.imwrite('../outputs/full_outputs/' + sample_names[i] + '.tiff', outputs[i])

#Save results from the model that had the best val_loss 
model = keras.models.load_model('../models/Current_Best.h5')
pred = model.predict(x_test)
pred_imgs = np.argmax(pred, axis=3)
outputs = restitch_images(sample_names, region_names, pred_imgs)
for i in range(len(outputs)):
	cv2.imwrite('../outputs/best_outputs/' + sample_names[i] + '.tiff', outputs[i])
