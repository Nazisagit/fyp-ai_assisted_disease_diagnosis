# Filename: data_collector.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Opened 23/02/2019

import csv
import os
from source.feature_detection.image_extractor import ImageExtractor
from source.feature_detection.FeatureDetector import FeatureDetector

""" This module collects data on the features detected 
	from the extracted images.
	The module will collect the following types of data:
	Width, Height, Area, Colour (split into R, G, B), and Length.
"""


def __init_output_files(data_output, group):
	output = data_output + group
	if not os.path.exists(output):
		os.makedirs(output)
	features = ['width', 'height', 'area', 'colour', 'length']
	for feature in features:
		feature_file = open(output + feature + '.csv', 'w', newline='')
		file_writer = csv.writer(feature_file)
		file_writer.writerow([feature.capitalize()])


def __save_data(feature_tables, data_output, group):
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
				elif feature == 1 and table[row][feature] > 0:
					height_list.append(table[row][feature])
				elif feature == 2 and table[row][feature] > 0:
					area_list.append(table[row][feature])
				elif feature == 3 and len(table[row][feature]) > 0:
					red_list.append(table[row][feature][0])
					blue_list.append(table[row][feature][1])
					green_list.append(table[row][feature][2])
				elif feature == 4 and table[row][feature] > 0:
					length_list.append(table[row][feature])

		colours = [red_list, green_list, blue_list]
		features = [width_list, height_list, area_list, length_list]
		files = ['width.csv', 'height.csv', 'area.csv', 'length.csv']
		for i in range(3):
			__save_feature(features[i], files[i], data_output, group)
		__save_colours(colours, 'colour.csv', data_output, group)


def __save_feature(feature, file, data_output, group):
	output = data_output + group
	with open(output + file, 'a', newline='') as output_file:
		file_writer = csv.writer(output_file)
		for measurement in feature:
			file_writer.writerow([str(measurement)])


def __save_colours(colours, file, data_output, group):
	output = data_output + group
	with open(output + file, 'a', newline='') as output_file:
		file_writer = csv.writer(output_file)
		for row in colours:
			file_writer.writerow([str(row[0]), str(row[1]), str(row[2])])


def __extract_images(original_images, extracted_images):
	if not os.path.exists(extracted_images):
		os.makedirs(extracted_images)
	image_extractor = ImageExtractor(original_images, extracted_images)
	image_extractor.extract()


def __detect_features(extracted_images, detected_features):
	if not os.path.exists(detected_features):
		os.makedirs(detected_features)
	feature_detector = FeatureDetector(extracted_images, detected_features)
	feature_detector.run()
	return feature_detector.get_feature_tables()


def __check_init_files(data_output, group):
	files = ['width.csv', 'height.csv', 'colour.csv', 'area.csv', 'length.csv']
	exists = []
	for file in files:
		path = data_output + group + file
		exists = [True if os.path.exists(path) else False]
	return all(exists)


def collect(input_dir, patient_number, patient_date, data_output, group):
	# directory of the original images
	original_images = input_dir + patient_number + '/' + patient_date + '/'
	# directory of the extracted images
	extracted_images = input_dir + 'extracted_images/' + patient_number + '/' + patient_date + '/'
	# directory of the feature detected images
	detected_features = input_dir + 'detected_features/' + patient_number + '/' + patient_date + '/'

	__extract_images(original_images, extracted_images)
	feature_tables = __detect_features(extracted_images, detected_features)

	if not __check_init_files(data_output, group):
		__init_output_files(data_output, group)
	__save_data(feature_tables, data_output, group)


if __name__ == '__main__':
	input_dir = '../../Student Data/'
	patient_number = '00099053314d'
	patient_date = '2017-06-12'
	data_output = '../../data_output/'
	group1 = 'group1/'
	group2 = 'group2/'
	group3 = 'group3/'
	collect(input_dir, patient_number, patient_date, data_output, group3)
