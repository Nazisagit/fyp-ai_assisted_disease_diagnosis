# Filename: data_collector.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Last modified: 06/04/2019

"""
Saves the detected features into separate csv files
"""

import csv
import os
from source.common.common import extract_images, detect_features


def __init_output_files(data_output, group):
	"""
	Initialises the output csv files
	:param data_output: folder where the IPCL group folders should be
	:param group: folder where the csv feature files should be initialised
	"""
	output = data_output + group
	if not os.path.exists(output):
		os.makedirs(output)
	features = ['width', 'height', 'area', 'length']
	for feature in features:
		feature_file = open(output + feature + '.csv', 'w', newline='')
		file_writer = csv.writer(feature_file)
		file_writer.writerow([feature.capitalize()])

	colour_file = open(output + 'colour.csv', 'w', newline='')
	colour_writer = csv.writer(colour_file)
	colour_writer.writerow(['Red', 'Green', 'Blue'])


def __save_data(feature_tables, data_output, group):
	"""
	Saves the data from the endoscopic images
	:param feature_tables: all the features tables from one patient
	:param data_output: folder where the IPCL group folders should be
	:param group: folder where the csv feature files should be
	"""
	width_list = list()
	height_list = list()
	area_list = list()
	red_list = list()
	green_list = list()
	blue_list = list()
	length_list = list()

	for table in feature_tables:
		table_height = len(table)
		for feature in [0, 1, 2, 3, 4]:
			for row in range(0, table_height):
				if feature == 0:
					width_list.append(table[row][feature])
				elif feature == 1:
					height_list.append(table[row][feature])
				elif feature == 2:
					area_list.append(table[row][feature])
				elif feature == 3:
					red_list.append(table[row][feature][0])
					blue_list.append(table[row][feature][1])
					green_list.append(table[row][feature][2])
				elif feature == 4:
					length_list.append(table[row][feature])

		colours = zip(red_list, green_list, blue_list)
		features = [width_list, height_list, area_list, length_list]
		files = ['width.csv', 'height.csv', 'area.csv', 'length.csv']
		for i in range(4):
			__save_feature(features[i], files[i], data_output, group)
		__save_colours(colours, data_output, group)


def __save_feature(feature, file, data_output, group):
	"""
	Saves all the features except colour
	:param feature: either width, height, area, or length
	:param file: the corresponding feature file
	:param data_output: folder where the IPCL group folders should be
	:param group: folder where the csv feature files should be
	"""
	output = data_output + group
	with open(output + file, 'a', newline='') as output_file:
		file_writer = csv.writer(output_file)
		for measurement in feature:
			file_writer.writerow([str(measurement)])


def __save_colours(colours, data_output, group):
	"""
	Saves the red, green, and blue colours of an IPCL
	:param colours: zip of lists of red, green, and blue
	:param data_output: folder where the IPCL group folders should be
	:param group: folder where the csv feature files should be
	"""
	output = data_output + group
	with open(output + 'colour.csv', 'a', newline='') as output_file:
		file_writer = csv.writer(output_file)
		for red, green, blue in colours:
			file_writer.writerow([str(red), str(green), str(blue)])


def __check_init_files(data_output, group):
	"""
	Checks to see if the csv feature files have been initialised
	:param data_output: folder where the IPCL group folders should be
	:param group: folder where the csv feature files should have been initialised
	:return: True if all the files exist
	"""
	files = ['width.csv', 'height.csv', 'area.csv', 'colour.csv', 'length.csv']
	exists = list()
	for file in files:
		path = data_output + group + file
		exists = [True if os.path.exists(path) else False]
	return all(exists)


def collect(images, patient_number, patient_date, data_output, group):
	"""
	Collects and saves the data from a patient's endoscopic images
	:param images: folder where all the image folders should be
	:param patient_number: folder of a patient
	:param patient_date: folder of the endoscopic images on the date taken
	:param data_output: folder of the outputted group folders
	:param group: folder where all the csv feature files should be
	"""
	# directory of the original images
	original_images = images + patient_number + '/' + patient_date + '/'
	# directory of the extracted images
	extracted_images = images + 'extracted_images/' + patient_number + '/' + patient_date + '/'
	# directory of the feature detected images
	detected_features = images + 'detected_features/' + patient_number + '/' + patient_date + '/'

	# extract the images
	extract_images(original_images, extracted_images)
	# further extract images
	# detect the features and save them to feature tables
	feature_tables = detect_features(extracted_images, detected_features)

	if not __check_init_files(data_output, group):
		__init_output_files(data_output, group)
	__save_data(feature_tables, data_output, group)


if __name__ == '__main__':
	# Provide the full path to folder of images
	images = 'D:/University/FYP/fyp-ai_assisted_disease_diagnosis/images/'
	# Provide the patient ID
	patient_number = '0017157764d'
	# Provide the date of the images
	patient_date = '2018-04-28'
	# Provide the full path to the data output folder
	data_output = 'D:/University/FYP/fyp-ai_assisted_disease_diagnosis/data_output/'
	# Choose your group
	group1 = 'group1/'
	group2 = 'group2/'
	group3 = 'group3/'
	collect(images, patient_number, patient_date, data_output, group3)
