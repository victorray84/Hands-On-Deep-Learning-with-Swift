#!/usr/bin/python
from __future__ import print_function
import os, sys, getopt
import shutil
import numpy as np
from scipy.misc import imresize, imsave
import matplotlib.pyplot as plt

TARGET_SIZE = (256,256)

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

	for sketch_dir, full_sketch_dir in get_sub_directories(source_path):
		# remove directory if already exists 
		if os.path.exists(os.path.join(dest_path, sketch_dir)):
			shutil.rmtree(os.path.join(dest_path, sketch_dir), ignore_errors=True)

		# create dest directory 
		os.mkdir(os.path.join(dest_path, sketch_dir))

		# iterate through all files in sketch directory 
		for sketch_filename, sketch_file_path in get_files(full_sketch_dir):
			img = plt.imread(sketch_file_path)
			image_scale = float(TARGET_SIZE[0]) / float(img.shape[0])
			rescaled_img = imresize(img, image_scale)

			save_file_path = os.path.join(os.path.join(dest_path, sketch_dir), sketch_filename)
			imsave(save_file_path, rescaled_img, format='png')

	print("Finished processing images; result in {}".format(dest_path))
			

def main(argv):
	source_path = ''
	dest_path = '' 

	try:
		opts, args = getopt.getopt(argv,"hs:d:",["src=","dst="])
	except getopt.GetoptError:
		print('preprocess_sketch_images.py -s <src directory> -d <dst directory>')
		sys.exit(2)
   
	for opt, arg in opts:
		if opt == '-h':
			print('preprocess_sketch_images.py -s <src directory> -d <dst directory>') 
			sys.exit()
		elif opt in ("-s", "--src"):
			source_path = arg.strip()
		elif opt in ("-d", "--dst"):
			dest_path = arg.strip()

	preprocess_images(source_path, dest_path)

if __name__ == "__main__":
	main(sys.argv[1:])