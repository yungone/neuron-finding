#! /usr/bin/python3

import json
import os
import numpy as np
import cv2

def unzip_dirs():
	base = '../../data/'
	dirs = os.listdir(base)
	for dir_name in dirs:
		os.system('unzip ' + base + dir_name + ' -d ' + base)

def avg_frames_training():
	base = '../../data/'
	dirs = os.listdir(base)
	for dir_name in dirs:
		frames = os.listdir(base + dir_name + '/images/')
		first_frame = cv2.imread(base + dir_name + '/images/' + frames[0])
		height = first_frame.shape[0]
		width = first_frame.shape[1]
		output = np.zeros((height, width))	
		for frame in frames:
			temp_frame = cv2.imread(base + dir_name + '/images/' + frame)
			output += temp_frame[:,:,0] / len(frames)
		np.save('../data/training/full/avg_frames/' + dir_name +  '.npy', output)

def create_masks_training():
	base = '../../data/'
	dirs = os.listdir(base)
	for dir_name in dirs:
		mask = np.zeros((512, 512))
		regions = json.load(open(base + dir_name + '/regions/regions.json'))
		for region in regions:
			for pair in region['coordinates']:
				mask[pair[0]][pair[1]] = 1
		np.save('../data/training/full/masks/' + dir_name + '.npy', mask)


		
def crop_avg_frames_training():
	base = '../data/training/full/avg_frames/'
	samples = os.listdir(base)
	for sample in samples:
		crop_counter = 0
		frame = np.load(base + sample)
		for i in range(1, 9):
			for j in range(1, 9):
				current = frame[64 * (i - 1): 64 * i, 64 * (j - 1): 64 * j]
				np.save('../data/training/cropped/avg_frames/' + \
					sample[: len(sample) - 4] + '-' + \
					str(crop_counter) + '.npy', \
					current)
				crop_counter += 1
def crop_avg_masks_training():
	base = '../data/training/full/masks/'
	samples = os.listdir(base)
	for sample in samples:
		crop_counter = 0
		mask = np.load(base + sample)
		for i in range(1, 9):
			for j in range(1, 9):
				current = mask[64 * (i - 1): 64 * i, 64 * (j - 1): 64 * j]
				np.save('../data/training/cropped/masks/' + \
					sample[: len(sample) - 4] + '-' + \
					str(crop_counter) + '.npy', \
					current)
				crop_counter += 1

#Testing
def avg_frames_testing():
	base = '../../data/'
	dirs = os.listdir(base)
	for dir_name in dirs:
		frames = os.listdir(base + dir_name + '/images/')
		first_frame = cv2.imread(base + dir_name + '/images/' + frames[0])
		height = first_frame.shape[0]
		width = first_frame.shape[1]
		output = np.zeros((height, width))	
		for frame in frames:
			temp_frame = cv2.imread(base + dir_name + '/images/' + frame)
			output += temp_frame[:,:,0] / len(frames)
		np.save('../data/testing/full/avg_frames/' + dir_name +  '.npy', output)

		
def crop_avg_frames_testing():
	base = '../data/testing/full/avg_frames/'
	samples = os.listdir(base)
	for sample in samples:
		crop_counter = 0
		frame = np.load(base + sample)
		for i in range(1, 9):
			for j in range(1, 9):
				current = frame[64 * (i - 1): 64 * i, 64 * (j - 1): 64 * j]
				np.save('../data/testing/cropped/avg_frames/' + \
					sample[: len(sample) - 4] + '-' + \
					str(crop_counter) + '.npy', \
					current)
				crop_counter += 1

#unzip_dirs()
#avg_frames_training()
#create_masks_training()
#crop_avg_frames_training()
#crop_avg_masks_training()
#avg_frames_testing()
#crop_avg_frames_testing()
