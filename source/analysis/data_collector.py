# Filename: data_collector.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Opened 23/02/2019

import csv
import os
from source.feature_detection.image_extractor import ImageExtractor
from source.feature_detection.feature_detector import FeatureDetector


class DataCollector:

	def __init__(self, feature_tables, data_output, ipcl_group):
		self.feature_tables = feature_tables
		self.data_output = data_output
		self.ipcl_group = ipcl_group

	def init_output_files(self):
		output = self.data_output + self.ipcl_group
		if not os.path.exists(output):
			os.makedirs(output)
		features = ['width', 'height', 'area', 'colour', 'length']
		for feature in features:
			feature_file = open(output + feature + '.csv', 'w', newline='')
			file_writer = csv.writer(feature_file)
			file_writer.writerow([feature.capitalize()])

	def save_data(self):
		width_list = list()
		height_list = list()
		area_list = list()
		colour_list = list()
		length_list = list()

		for table in self.feature_tables:
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
						colour_list.append(table[row][feature])
					elif feature == 4 and table[row][feature] > 0:
						length_list.append(table[row][feature])

			features = [width_list, height_list, area_list, colour_list, length_list]
			files = ['width.csv', 'height.csv', 'area.csv', 'colour.csv', 'length.csv']
			for i in range(4):
				self.__save_feature(features[i], files[i], self.ipcl_group)

	def __save_feature(self, feature, file, group):
		output = self.data_output + group
		with open(output + file, 'a', newline='') as output_file:
			file_writer = csv.writer(output_file)
			for measurement in feature:
				file_writer.writerow([str(measurement)])


def extract_images(original_images, extracted_images):
	if not os.path.exists(extracted_images):
		os.makedirs(extracted_images)
	image_extractor = ImageExtractor(original_images, extracted_images)
	image_extractor.extract()


def detect_features(extracted_images, detected_features):
	if not os.path.exists(detected_features):
		os.makedirs(detected_features)
	feature_detector = FeatureDetector(extracted_images, detected_features)
	feature_detector.run()
	return feature_detector.get_feature_tables()


def check_init_files(data_output, group):
	files = ['width.csv', 'height.csv', 'colour.csv', 'area.csv', 'length.csv']
	exists = []
	for file in files:
		path = data_output + group + file
		exists = [True if os.path.exists(path) else False]
	return all(exists)


def collect(images, patient_number, patient_date, data_output, group):
	original_images = images + patient_number + '/' + patient_date + '/'
	# directory of the extracted images
	extracted_images = images + 'extracted_images/' + patient_number + '/' + patient_date + '/'
	# directory of the feature detected images
	detected_features = images + 'detected_features/' + patient_number + '/' + patient_date + '/'

	extract_images(original_images, extracted_images)
	feature_tables = detect_features(extracted_images, detected_features)

	data_collector = DataCollector(feature_tables, data_output, group)
	# Comment out this command after collecting data for your first image
	if not check_init_files(data_output, group):
		data_collector.init_output_files()
	data_collector.save_data()


if __name__ == '__main__':
	group1 = 'group1/'
	group2 = 'group2/'
	group3 = 'group3/'
	collect('../../Student Data/', '0099053314d', '2017-06-12', '../../data_output2/', group3)
