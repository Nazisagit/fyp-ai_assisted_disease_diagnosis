# Filename: data_collector.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Opened 23/02/2019

import csv
import os
from source.feature_detection.image_extractor import ImageExtractor
from source.feature_detection.FeatureDetector import FeatureDetector
from source.feature_detection.FurtherExtractor import FurtherExtractor

""" This module collects data on the features detected 
	from the extracted images.
	The module will collect the following types of data:
	Width, Height, Area, Colour (R, G, B).
"""


def __init_output_files(data_output, group):
	output = data_output + group
	if not os.path.exists(output):
		os.makedirs(output)
	features = ['width', 'height', 'area']
	for feature in features:
		feature_file = open(output + feature + '.csv', 'w', newline='')
		file_writer = csv.writer(feature_file)
		file_writer.writerow([feature.capitalize()])

	colour_file = open(output + 'colour.csv', 'w', newline='')
	colour_writer = csv.writer(colour_file)
	colour_writer.writerow(['Red', 'Green', 'Blue'])


def __save_data(feature_tables, data_output, group):
	width_list = list()
	height_list = list()
	area_list = list()
	red_list = list()
	blue_list = list()
	green_list = list()

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
					green_list.append(table[row][feature][1])
					blue_list.append(table[row][feature][2])

		colours = [red_list, green_list, blue_list]
		features = [width_list, height_list, area_list]
		files = ['width.csv', 'height.csv', 'area.csv']
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


def __further_extract(extracted_images, further_extracted_images):
	if not os.path.exists(further_extracted_images):
		os.makedirs(further_extracted_images)
	further_extractor = FurtherExtractor(extracted_images, further_extracted_images)
	further_extractor.run()


def __detect_features(extracted_images, detected_features):
	if not os.path.exists(detected_features):
		os.makedirs(detected_features)
	feature_detector = FeatureDetector(extracted_images, detected_features)
	feature_detector.run()
	return feature_detector.get_feature_tables()


def __check_init_files(data_output, group):
	files = ['width.csv', 'height.csv', 'colour.csv', 'area.csv']
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
	# directory of further extracted images
	further_extracted_images = input_dir + 'further_extracted/' + patient_number + '/' + patient_date + '/'
	# directory of the feature detected images
	detected_features = input_dir + 'detected_features/' + patient_number + '/' + patient_date + '/'

	__extract_images(original_images, extracted_images)
	__further_extract(extracted_images, further_extracted_images)
	feature_tables = __detect_features(further_extracted_images, detected_features)

	if __check_init_files(data_output, group) is False:
		__init_output_files(data_output, group)
	__save_data(feature_tables, data_output, group)


if __name__ == '__main__':
	input_dir = '../../Student Data/'
	patient_number = '0096043466d'
	patient_date = '2018-07-06'
	data_output = '../../data_output-further/'
	group1 = 'group1/'
	group2 = 'group2/'
	group3 = 'group3/'
	collect(input_dir, patient_number, patient_date, data_output, group3)
