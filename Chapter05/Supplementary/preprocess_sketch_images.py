#!/usr/bin/python

"""

"""

from __future__ import print_function
import os, sys, getopt
import shutil
import random 
import requests 
import zipfile
import numpy as np
from scipy.misc import imresize, imsave
from scipy import ndimage
import matplotlib.pyplot as plt

SKETCHES_URL = 'http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip'

CATEGORIES = [
	'airplane', 
	'apple', 
	'bear (animal)', 
	'bee', 
	'bicycle', 
	'butterfly', 
	'car (sedan)', 
	'cat', 
	'chair', 
	'cup', 
	'dog', 
	'elephant', 
	'head', 
	'helicopter', 
	'monkey', 
	'owl', 
	'panda', 
	'penguin', 
	'pigeon', 
	'rabbit', 
	'sailboat', 
	'teddy-bear'
	]

TARGET_SIZE = (128,128)

BACKGROUND_COLOR = 0

INVERT_COLOR = True 

def is_image(file_path):
	image_extensions = ['png', 'jpg', 'jpeg']

	for image_extension in image_extensions:
		if image_extension in file_path.lower():
			return True 

	return False 		

def get_sub_directories(full_path):
	
	for d in os.listdir(full_path):				
		if not os.path.isdir(os.path.join(full_path, d)):
			continue 		

		yield d, os.path.join(full_path, d) 

def get_files(dir_path):
	for f in os.listdir(dir_path):
		full_path = os.path.join(dir_path, f) 		
		if os.path.isfile(full_path) and is_image(full_path):
			yield f, full_path

def preprocess_images(source_path, dest_path):	
	print("preprocessing image from {}".format(source_path))

	# create train and valid directories 
	if os.path.exists(os.path.join(dest_path, 'train')):
		shutil.rmtree(os.path.join(dest_path, 'train'), ignore_errors=True)

	os.mkdir(os.path.join(dest_path, 'train'))

	if os.path.exists(os.path.join(dest_path, 'valid')):
		shutil.rmtree(os.path.join(dest_path, 'valid'), ignore_errors=True)

	os.mkdir(os.path.join(dest_path, 'valid'))

	for sketch_dir, full_sketch_dir in get_sub_directories(source_path):
		train_sketch_dir = os.path.join('train', sketch_dir)
		valid_sketch_dir = os.path.join('valid', sketch_dir)

		# remove directory if already exists 
		if os.path.exists(os.path.join(dest_path, train_sketch_dir)):
			shutil.rmtree(os.path.join(dest_path, train_sketch_dir), ignore_errors=True)

		if os.path.exists(os.path.join(dest_path, valid_sketch_dir)):
			shutil.rmtree(os.path.join(dest_path, valid_sketch_dir), ignore_errors=True)

		# create dest directory 				
		os.mkdir(os.path.join(dest_path, train_sketch_dir))
		os.mkdir(os.path.join(dest_path, valid_sketch_dir))

		total_count = sum([1 for _ in get_files(full_sketch_dir)]) 
		valid_count = total_count * 0.1 # leave out 10% for validation 
		current_index = 0 		

		# iterate through all files in sketch directory 
		for sketch_filename, sketch_file_path in get_files(full_sketch_dir):
			img = plt.imread(sketch_file_path)
			#img /= 255.
			if INVERT_COLOR:
				img = 255.0 - img
            	
			image_scale = float(TARGET_SIZE[0]) / float(img.shape[0])
			rescaled_img = imresize(img, image_scale)

			if current_index < valid_count:				
				save_file_path = os.path.join(os.path.join(dest_path, valid_sketch_dir), sketch_filename)
			else:
				save_file_path = os.path.join(os.path.join(dest_path, train_sketch_dir), sketch_filename)										

			imsave(save_file_path, rescaled_img, format='png')

			if current_index >= valid_count:
				# image augmentation 

				# horiziontal flip  
				if random.random() > 0.4:
					save_file_path = os.path.join(os.path.join(dest_path, train_sketch_dir), "f_" + sketch_filename)
					flipped_img = np.fliplr(rescaled_img)
					imsave(save_file_path, flipped_img, format='png')					

				# translate
				translated = False
				translated_img = np.copy(rescaled_img)

				if random.random() > 0.8:
					translated_img = shift_left(translated_img)
					translated = True

				if random.random() > 0.8:
					translated_img = shift_right(translated_img)
					translated = True

				if random.random() > 0.8:
					translated_img = shift_up(translated_img)
					translated = True

				if translated:
					save_file_path = os.path.join(os.path.join(dest_path, train_sketch_dir), "t_" + sketch_filename)
					imsave(save_file_path, translated_img, format='png')

				# rotate
				if random.random() > 0.8:
					if random.random() > 0.5:
						rotated_img = ndimage.rotate(rescaled_img, 15, reshape=False)
					else:
						rotated_img = ndimage.rotate(rescaled_img, -15, reshape=False)

					save_file_path = os.path.join(os.path.join(dest_path, train_sketch_dir), "r_" + sketch_filename)
					imsave(save_file_path, rotated_img, format='png')

			current_index += 1

	print("Finished processing images; result in {}".format(dest_path))

def shift_left(img, ox=5):
	HEIGHT, WIDTH = img.shape[0], img.shape[1]

	for i in range(img.shape[0], 1, -1):
		for j in range(img.shape[1]):
			if (i < img.shape[0]-ox):
				img[j][i] = img[j][i-ox]
			elif (i < img.shape[0]-1):
				img[j][i] = BACKGROUND_COLOR
	return img 

def shift_right(img, ox=5):
	HEIGHT, WIDTH = img.shape[0], img.shape[1]

	for j in range(WIDTH):
		for i in range(HEIGHT):
			if (i < HEIGHT-ox):
				img[j][i] = img[j][i+ox]

	return img 

def shift_up(img, oy=5):
	HEIGHT, WIDTH = img.shape[0], img.shape[1]

	for j in range(WIDTH):
		for i in range(HEIGHT):
			if (j < WIDTH - oy and j > oy):
				img[j][i] = img[j+oy][i]
			else:
				img[j][i] = BACKGROUND_COLOR
	
	return img 

def download_and_extract_files(url, dest_path):
	"""
	Download and extract files and return the extracted path 
	"""

	download_folder = os.path.join(dest_path, 'org')
	download_file = os.path.join(download_folder, 'sketches.zip')

	# create train and valid directories 
	if os.path.exists(download_folder):
		shutil.rmtree(download_folder, ignore_errors=True)

	os.mkdir(download_folder)

	r = requests.get(url)
	with open(download_file,'wb') as f: 
		f.write(r.content) 

	with zipfile.ZipFile(download_file) as zip_f:
		zip_f.extractall(download_folder)

	# delete zip 
	os.remove(download_file)

	extracted_folder = os.path.join(download_folder, 'png')
	return extracted_folder

def filter_categories(extracted_sketches_folder):
	"""
	Remove any cateogies that are not present in the CATEGORIES list 
	"""
	folders = [folder[0] for folder in get_sub_directories(extracted_sketches_folder)]

	for folder in folders:
		if folder not in CATEGORIES:
			shutil.rmtree(os.path.join(extracted_sketches_folder, folder), ignore_errors=True)		
			

def main(argv):		
	dest_path = ''	

	try:
		opts, args = getopt.getopt(argv,"hd:",["dst="])
	except getopt.GetoptError:
		print('preprocess_sketch_images.py -d <dst directory>')
		sys.exit(2)
   
	for opt, arg in opts:
		if opt == '-h':
			print('preprocess_sketch_images.py -d <dst directory>') 
			sys.exit()
		elif opt in ("-d", "--dst"):
			dest_path = arg.strip()
			print(dest_path) 

	if len(dest_path) == 0:
		print('preprocess_sketch_images.py -d <dst directory>')
		sys.exit(2)

	extracted_sketches_folder = download_and_extract_files(SKETCHES_URL, dest_path)

	filter_categories(extracted_sketches_folder)
	
	preprocess_images(extracted_sketches_folder, dest_path)

if __name__ == "__main__":

	main(sys.argv[1:])
