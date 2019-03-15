# Filename: DataProcessing.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Opened 23/02/2019

import pandas as pd
import csv
import os


class DataProcessing:

	@staticmethod
	def calculate_mean(csv_input):
		return pd.read_csv(csv_input).mean()

	@staticmethod
	def calculate_median(csv_input):
		return pd.read_csv(csv_input).median()

	@staticmethod
	def calculate_std(csv_input):
		return pd.read_csv(csv_input).std()

	@staticmethod
	def calculate_modes(csv_input):
		return pd.read_csv(csv_input).mode()


class DataCollecting:

	def __init__(self, feature_tables, number_ipcls, data_output, ipcl_type):
		self.feature_tables = feature_tables
		self.data_output = data_output
		self.ipcl_type = ipcl_type
		self.number_ipcls = number_ipcls

	def init_output_files(self):
		output = self.data_output + self.ipcl_type
		if not os.path.exists(output):
			os.makedirs(output)
		self.init_width_file(output)
		self.init_height_file(output)
		self.init_rotation_file(output)
		self.init_area_file(output)
		self.init_colour_file(output)
		self.init_length_file(output)
		self.init_occurrence_file(output)

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

	@staticmethod
	def init_occurrence_file(output):
		occurrences_file = open(output + 'occurrences.csv', 'w', newline='')
		occurrences_writer = csv.writer(occurrences_file)
		occurrences_writer.writerow(['Occurrences'])
		occurrences_file.close()

	def save_data(self):
		width_list = list()
		height_list = list()
		rotation_list = list()
		area_list = list()
		colour_list = list()
		length_list = list()

		for table in self.feature_tables:
			table_height = len(table)
			for feature in [0, 1, 2, 3, 4, 5]:
				for row in range(0, table_height):
					if feature == 0:
						width_list.append(table[row][feature])
					elif feature == 1 and table[row][feature] > 0:
						height_list.append(table[row][feature])
					elif feature == 2 and table[row][feature] > 0:
						rotation_list.append(abs(table[row][feature]))
					elif feature == 3 and table[row][feature] > 0:
						area_list.append(table[row][feature])
					elif feature == 4 and len(table[row][feature]) > 0:
						colour_list.append(table[row][feature])
					elif feature == 5 and table[row][feature] > 0:
						length_list.append(table[row][feature])

			self.save_width(width_list, self.ipcl_type)
			self.save_height(height_list, self.ipcl_type)
			self.save_rotation(rotation_list, self.ipcl_type)
			self.save_area(area_list, self.ipcl_type)
			self.save_colour(colour_list, self.ipcl_type)
			self.save_length(length_list, self.ipcl_type)
		self.save_occurrences(self.number_ipcls, self.ipcl_type)

	def save_width(self, width_list, ipcl_type):
		output = self.data_output + ipcl_type
		with open(output + 'width.csv', 'a', newline='') as width_file:
			width_writer = csv.writer(width_file)
			for width in width_list:
				width_writer.writerow([str(width)])

	def save_height(self, height_list, ipcl_type):
		output = self.data_output + ipcl_type
		with open(output + 'height.csv', 'a', newline='') as height_file:
			height_writer = csv.writer(height_file)
			for height in height_list:
				height_writer.writerow([str(height)])

	def save_rotation(self, rotation_list, ipcl_type):
		output = self.data_output + ipcl_type
		with open(output + 'rotation.csv', 'a', newline='') as rotation_file:
			rotation_writer = csv.writer(rotation_file)
			for rotation in rotation_list:
				rotation_writer.writerow([str(rotation)])

	def save_area(self, area_list, ipcl_type):
		output = self.data_output + ipcl_type
		with open(output + 'area.csv', 'a', newline='') as area_file:
			area_writer = csv.writer(area_file)
			for area in area_list:
				area_writer.writerow([str(area)])

	def save_colour(self, colour_list, ipcl_type):
		output = self.data_output + ipcl_type
		with open(output + 'colour.csv', 'a', newline='') as colour_file:
			colour_writer = csv.writer(colour_file)
			for colour in colour_list:
				colour_writer.writerow([str(colour[0]), str(colour[1]), str(colour[2])])

	def save_length(self, length_list, ipcl_type):
		output = self.data_output + ipcl_type
		with open(output + 'length.csv', 'a', newline='') as length_file:
			length_writer = csv.writer(length_file)
			for length in length_list:
				length_writer.writerow([str(length)])

	def save_occurrences(self, occurrences_list, ipcl_type):
		output = self.data_output + ipcl_type
		with open(output + 'occurrences.csv', 'a', newline='') as occurrences_file:
			occurrences_writer = csv.writer(occurrences_file)
			for occurrence in occurrences_list:
				occurrences_writer.writerow([str(occurrence)])


