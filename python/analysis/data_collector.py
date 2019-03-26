# Filename: data_collector.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Opened 23/02/2019

import csv
import os
from pathlib import Path
from python.feature_detection.image_extractor import ImageExtractor
from python.feature_detection.feature_detector import FeatureDetector


class DataCollector:

	def __init__(self, feature_tables, data_output, ipcl_group):
		self.feature_tables = feature_tables
		self.data_output = data_output
		self.ipcl_group = ipcl_group

	def init_output_files(self):
		output = self.data_output + self.ipcl_group
		if not os.path.exists(output):
			os.makedirs(output)
		self.init_width_file(output)
		self.init_height_file(output)
		self.init_rotation_file(output)
		self.init_area_file(output)
		self.init_colour_file(output)
		self.init_length_file(output)

	@staticmethod
	def init_width_file(output):
		width_file = open(output + 'width.csv', 'w', newline='')
		width_writer = csv.writer(width_file)
		width_writer.writerow(['Width'])
		width_file.close()

	@staticmethod
	def init_height_file(output):
		height_file = open(output + 'height.csv', 'w', newline='')
		height_writer = csv.writer(height_file)
		height_writer.writerow(['Height'])
		height_file.close()

	@staticmethod
	def init_rotation_file(output):
		rotation_file = open(output + 'rotation.csv', 'w', newline='')
		rotation_writer = csv.writer(rotation_file)
		rotation_writer.writerow(['Rotation'])
		rotation_file.close()

	@staticmethod
	def init_area_file(output):
		area_file = open(output + 'area.csv', 'w', newline='')
		area_writer = csv.writer(area_file)
		area_writer.writerow(['Area'])
		area_file.close()

	@staticmethod
	def init_colour_file(output):
		colour_file = open(output + 'colour.csv', 'w', newline='')
		colour_writer = csv.writer(colour_file)
		colour_writer.writerow(['Red', 'Green', 'Blue'])
		colour_file.close()

	@staticmethod
	def init_length_file(output):
		length_file = open(output + 'length.csv', 'w', newline='')
		length_writer = csv.writer(length_file)
		length_writer.writerow(['Length'])
		length_file.close()

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

			self.save_width(width_list, self.ipcl_group)
			self.save_height(height_list, self.ipcl_group)
			self.save_area(area_list, self.ipcl_group)
			self.save_colour(colour_list, self.ipcl_group)
			self.save_length(length_list, self.ipcl_group)

	def save_width(self, width_list, ipcl_group):
		output = self.data_output + ipcl_group
		with open(output + 'width.csv', 'a', newline='') as width_file:
			width_writer = csv.writer(width_file)
			for width in width_list:
				width_writer.writerow([str(width)])

	def save_height(self, height_list, ipcl_group):
		output = self.data_output + ipcl_group
		with open(output + 'height.csv', 'a', newline='') as height_file:
			height_writer = csv.writer(height_file)
			for height in height_list:
				height_writer.writerow([str(height)])

	def save_area(self, area_list, ipcl_group):
		output = self.data_output + ipcl_group
		with open(output + 'area.csv', 'a', newline='') as area_file:
			area_writer = csv.writer(area_file)
			for area in area_list:
				area_writer.writerow([str(area)])

	def save_colour(self, colour_list, ipcl_group):
		output = self.data_output + ipcl_group
		with open(output + 'colour.csv', 'a', newline='') as colour_file:
			colour_writer = csv.writer(colour_file)
			for colour in colour_list:
				colour_writer.writerow([str(colour[0]), str(colour[1]), str(colour[2])])

	def save_length(self, length_list, ipcl_group):
		output = self.data_output + ipcl_group
		with open(output + 'length.csv', 'a', newline='') as length_file:
			length_writer = csv.writer(length_file)
			for length in length_list:
				length_writer.writerow([str(length)])


def extract_images(original_images, extracted_images):
	path = Path(extracted_images)
	if not path.exists():
		path.mkdir()
	image_extractor = ImageExtractor(original_images, extracted_images)
	image_extractor.extract()


def detect_features(extracted_images, detected_features):
	path = Path(detected_features)
	if not path.exists():
		path.mkdir()
	feature_detector = FeatureDetector(extracted_images, detected_features)
	feature_detector.run()
	return feature_detector.get_feature_tables(), feature_detector.get_number_ipcls()


def check_init_files(data_output, group):
	width = Path(data_output + group + 'width.csv')
	height = Path(data_output + group + 'height.csv')
	colour = Path(data_output + group + 'colour.csv')
	area = Path(data_output + group + 'area.csv')
	length = Path(data_output + group + 'length.csv')
	paths = [width, height, colour, area, length]
	for path in paths:
		return True if path.exists() else False


def collect(images, patient_number, patient_date, data_output, group):
	original_images = images + patient_number + '/' + patient_date + '/'
	# directory of the extracted images
	extracted_images = images + '../extracted_images/' + patient_number + '/' + patient_date + '/'
	# directory of the feature detected images
	detected_features = images + '../detected_features/' + patient_number + '/' + patient_date + '/'

	extract_images(original_images, extracted_images)
	feature_tables, number_ipcls = detect_features(extracted_images, detected_features)

	data_collector = DataCollector(feature_tables, data_output, group)
	# Comment out this command after collecting data for your first image
	if check_init_files(data_output, group):
		data_collector.init_output_files()
	data_collector.save_data()


if __name__ == '__main__':
	group1 = 'group1/'
	group2 = 'group2/'
	group3 = 'group3/'
	collect('../Student Data/', '0099053314d', '2017-06-12', '../../data_output/', group3)
