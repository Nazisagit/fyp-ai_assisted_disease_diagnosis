# Filename: ImageExtractor
# Author: Nazrin Pengiran
# Institution: King's College London

import cv2
from os import listdir
from os.path import isfile, join
import numpy as np


class ImageExtractor:

	def __init__(self, images_path, output_folder):
		# Path to images
		self.images_path = images_path
		# Output folder path
		self.output_folder = output_folder

	# Crops the octagon shaped region that contains the video
	# footage / image from the rest of the elements of the frame

	def extract(self):
		files = [f for f in listdir(self.images_path) if isfile(join(self.images_path, f))]
		counter = 0

		for file in files:
			if file.endswith('.png') or file.endswith('.jpg'):
				image = cv2.imread(self.images_path + str(file))
				self.show_progressbar(counter, len(files))
				counter += 1
				if self.get_darkness_percent(image) < 0.3:
					cropped_image = self.crop_frame(image)
					# cv2.imwrite(self.output_folder + 'cropped' + str(file), cropped_image)
					########## 01/02/2019
					# Each divided image gets saved as cropped and divided
					divided_image = self.divide_image(cropped_image)
					for i in range(len(divided_image)):
						img = divided_image[i]
						final_output = self.output_folder + str(i) + '_cropped_and_divided_' + str(file)
						cv2.imwrite(final_output, img)
					#######################################################################################

	@staticmethod
	def crop_frame(frame):
		########## 03/02/2019
		## lower_black = np.array([0, 0, 16])
		## upper_black = np.array([165, 120, 23])
		# I use BGR instead of HSV values because the frame
		# does not get cropped using HSV
		lower_black = np.array([0, 0, 0])
		upper_black = np.array([0, 0, 0])
		# Threshold the HSV image to mark all black regions as foreground
		## mask = self.hsv_colour_threshold(frame, lower_value=lower_black, upper_value=upper_black)
		mask = cv2.inRange(frame, lower_black, upper_black)
		###################################################
		# Perform morphological closing to eliminate small objects like text and icons
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((100, 100), np.uint8))

		# Find the biggest blob...
		# Inverse the mask
		mask_inverse = 255 - mask

		# Identify the contours
		_, contours, _ = cv2.findContours(mask_inverse, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
	def get_darkness_percent(self, frame):
		# Define the range of white color in HSV
		lower_dark = np.array([0, 0, 30])
		upper_dark = np.array([30, 55, 37])

		# Threshold the HSV image to mark the glare parts as foreground
		mask = self.hsv_colour_threshold(frame, lower_value=lower_dark, upper_value=upper_dark)

		# Calculate frame darkness percentage
		height, width, channels = frame.shape
		darkness = cv2.countNonZero(mask) / (width * height)

		return darkness

	# Performs HSV colour threshold on the image based on the upper and
	# lower threshold values provided and returns the resulting mask.
	@staticmethod
	def hsv_colour_threshold(frame, lower_value, upper_value):
		# Convert RGB frame to HSV for better colour separation
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# Threshold the HSV image to mark all black regions as foreground
		mask = cv2.inRange(hsv, lower_value, upper_value)

		return mask

	# Prints the progress of video / image analysis to the console.
	@staticmethod
	def show_progressbar(iteration, total, fill='█'):
		# The code taken from:
		# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
		percent = ("{0:." + str(1) + "f}").format(100 * (iteration / float(total)))
		filled_length = int(50 * iteration // total)
		bar = fill * filled_length + '-' * (50 - filled_length)
		print('\r%s |%s| %s%% %s' % ('Progress', bar, percent, 'Complete'), end='\r')
		# Print New Line on Complete
		if iteration == total:
			print()

	########## 01/02/2019
	# Divides the image into 4 images
	@staticmethod
	def divide_image(frame):
		divided_images = list()
		imcol, imrow, _ = frame.shape
		colstep = int(imcol/2)
		rowstep = int(imrow/2)

		for i in range(0, imcol, colstep):
			for j in range(0, imrow, rowstep):
				block = frame[i:i + colstep, j:j + rowstep]
				blockcol, blockrow, _ = block.shape
				if blockcol > 1 and blockrow > 1:
					divided_images.append(block)
		return divided_images
	#########################
