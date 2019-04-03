# Filename: image_extractor.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Credits: Dmitry Poliyivets

import cv2
from os import listdir
from os.path import isfile, join
import numpy as np

"""
This module is comprised of python functions created by Dmitry Poliyivets 
from his final year project in his FrameExtractor.py file. 
The functions are all needed to extract region of interest
images. 
Some modifications have been made to pythonize the style of the code, and 
to work with .jpgs and .pngs instead of video.
"""


def extract(images_path, output_folder):
	files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
	counter = 0

	for file in files:
		if file.endswith('.png') or file.endswith('.jpg'):
			image = cv2.imread(images_path + str(file))
			__show_progressbar(counter, len(files))
			counter += 1
			if __get_darkness_percent(image) < 0.3:
				cropped_image = __crop_frame(image)
				final_output = output_folder + '_cropped_' + str(file)
				cv2.imwrite(final_output, cropped_image)


def __crop_frame(frame):
	lower_black = np.array([0, 0, 16])
	upper_black = np.array([165, 120, 23])
	# Threshold the HSV image to mark all black regions as foreground
	mask = __hsv_colour_threshold(frame, lower_value=lower_black, upper_value=upper_black)
	# Perform morphological closing to eliminate small objects like text and icons
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((100, 100), np.uint8))

	# Find the biggest blob...
	# Inverse the mask
	mask_inverse = 255 - mask

	# Identify the contours
	contours, _ = cv2.findContours(mask_inverse, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# Initialize necessary variables
	max_area, max_area_index = 0, 0
	# Loop over the contours
	for index, c in enumerate(contours):
		# Get area of the current blob
		area = cv2.contourArea(c)
		# Check if the area of current blob is greater than previous maxArea
		if area > max_area:
			# Record this area as maxArea and record its index
			max_area = area
			max_area_index = index

	# Identify the bounding box around the largest blob
	x, y, w, h = cv2.boundingRect(contours[max_area_index])

	# Crop input frame according to the side of the bounding box
	cropped_frame = frame[y:y + h, x:x + w]

	# Return the cropped frame
	return cropped_frame


# Identifies the percentage of the frame that is very dark.
# Used to make sure no dark frames are saved to disk.
def __get_darkness_percent(frame):
	lower_dark = np.array([0, 0, 30])
	upper_dark = np.array([30, 55, 37])

	mask = __hsv_colour_threshold(frame, lower_value=lower_dark, upper_value=upper_dark)

	# Calculate frame darkness percentage
	height, width, channels = frame.shape
	darkness = cv2.countNonZero(mask) / (width * height)

	return darkness


# Performs HSV colour threshold on the image based on the upper and
# lower threshold values provided and returns the resulting mask.
def __hsv_colour_threshold(frame, lower_value, upper_value):
	# Convert RGB frame to HSV for better colour separation
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Threshold the HSV image to mark all black regions as foreground
	mask = cv2.inRange(hsv, lower_value, upper_value)

	return mask


# Prints the progress of video / image analysis to the console.
def __show_progressbar(iteration, total, fill='█'):
	# The code taken from:
	# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
	percent = ("{0:." + str(1) + "f}").format(100 * (iteration / float(total)))
	filled_length = int(50 * iteration // total)
	bar = fill * filled_length + '-' * (50 - filled_length)
	print('\r%s |%s| %s%% %s' % ('Progress', bar, percent, 'Complete'), end='\r')
	# Print New Line on Complete
	if iteration == total:
		print('\n')
