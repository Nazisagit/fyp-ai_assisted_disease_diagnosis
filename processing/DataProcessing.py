# Filename: DataProcessing.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Opened 23/02/2019

import numpy as np
import pandas as pd
from collections import Counter
import csv
import os


class DataProcessing:

	@staticmethod
	def max_min_csv(data_input):
		cols = ['Max', 'Min']
		data = pd.read_csv(data_input, usecols=cols)
		return max(data.Max), min(data.Min)


class DataCollecting:

	def __init__(self, feature_tables, data_output):
		self.feature_tables = feature_tables
		self.data_output = data_output

	def init_output_files(self, ipcl_type):
		output = self.data_output + ipcl_type
		if not os.path.exists(output):
			os.makedirs(output)
		self.init_rotation_file(output)
		self.init_area_file(output)
		self.init_length_file(output)
		self.init_width_file(output)
		self.init_height_file(output)
		self.init_calibre_file(output)

	@staticmethod
	def init_rotation_file(output):
		rotation_file = open(output + 'rotation.csv', 'w', newline='')
		rotation_writer = csv.writer(rotation_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
		rotation_writer.writerow(('Max', 'Min'))
		rotation_file.close()

	@staticmethod
	def init_area_file(output):
		area_file = open(output + 'area.csv', 'w', newline='')
		area_writer = csv.writer(area_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
		area_writer.writerow(('Max', 'Min'))
		area_file.close()

	@staticmethod
	def init_length_file(output):
		length_file = open(output + 'length.csv', 'w', newline='')
		length_writer = csv.writer(length_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
		length_writer.writerow(('Max', 'Min'))
		length_file.close()

	@staticmethod
	def init_width_file(output):
		width_file = open(output + 'width.csv', 'w', newline='')
		width_writer = csv.writer(width_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
		width_writer.writerow(('Max', 'Min'))
		width_file.close()

	@staticmethod
	def init_height_file(output):
		height_file = open(output + 'height.csv', 'w', newline='')
		height_writer = csv.writer(height_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
		height_writer.writerow(('Max', 'Min'))
		height_file.close()

	@staticmethod
	def init_calibre_file(output):
		calibre_file = open(output + 'calibre.csv', 'w', newline='')
		calibre_writer = csv.writer(calibre_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
		calibre_writer.writerow(('Max', 'Min'))
		calibre_file.close()

	def save_data(self):
		rotation_set = set()
		area_set = set()
		length_set = set()
		width_set = set()
		height_set = set()

		for table in self.feature_tables:
			table_height = len(table)
			# For Rotation, Area, and Length
			for feature in [0, 1, 2, 3, 4]:
				# For all entries in the table
				for row in range(0, table_height):
					# Calculate Total Rotation
					if feature == 0:
						rotation_set.add(table[row][feature])
					elif feature == 1:
						area_set.add(table[row][feature])
					elif feature == 2:
						length_set.add(table[row][feature])
					# Calculate Total Area
					elif feature == 3:
						width_set.add(table[row][feature])
					# Calculate Total Length
					elif feature == 4:
						height_set.add(table[row][feature])
			calibre_set = self.calculate_calibres(area_set, length_set)
			self.save_rotation(max(rotation_set), min(rotation_set), 'type1/')
			self.save_area(max(area_set), min(area_set), 'type1/')
			self.save_length(max(length_set), min(length_set), 'type1/')
			self.save_width(max(width_set), min(width_set), 'type1/')
			self.save_height(max(height_set), min(height_set), 'type1/')
			self.save_calibre(max(calibre_set), min(calibre_set), 'type1/')

	@staticmethod
	def calculate_calibres(area_vector, length_vector):
		calibre_vector = list()
		for area in area_vector:
			for length in length_vector:
				calibre_vector.append(area / length)
		return calibre_vector

	@staticmethod
	def calculate_means(vectors):
		means = list()
		for vector in vectors:
			means.append(np.mean(vector))
		return means

	@staticmethod
	def calculate_medians(vectors):
		medians = list()
		for vector in vectors:
			medians.append(np.median(vector))
		return medians

	@staticmethod
	def calculate_stds(vectors):
		stds = list()
		for vector in vectors:
			stds.append(np.std(vector, dtype=np.float64))
		return stds

	def calculate_modes(self, vectors):
		modes = list()
		for vector in vectors:
			modes.append(self.get_mode(vector))
		return modes

	@staticmethod
	def get_mode(vector):
		counter = Counter(vector)
		if len(counter.most_common(1)) >= 1:
			_, val = counter.most_common(1)[0]
			return [x for x, y in counter.items() if y == val]
		else:
			return []

	def save_rotation(self, max_rotation, min_rotation, ipcl_type):
		output = self.data_output + ipcl_type
		with open(output + 'rotation.csv', 'a', newline='') as rotation_file:
			rotation_writer = csv.writer(rotation_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			rotation_writer.writerow((str(max_rotation), str(min_rotation)))

	def save_area(self, max_area, min_area, ipcl_type):
		output = self.data_output + ipcl_type
		with open(output + 'area.csv', 'a', newline='') as area_file:
			area_writer = csv.writer(area_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			area_writer.writerow((str(max_area), str(min_area)))

	def save_length(self, max_length, min_length, ipcl_type):
		output = self.data_output + ipcl_type
		with open(output + 'length.csv', 'a', newline='') as length_file:
			length_writer = csv.writer(length_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			length_writer.writerow((str(max_length), str(min_length)))

	def save_width(self, max_width, min_width, ipcl_type):
		output = self.data_output + ipcl_type
		with open(output + 'width.csv', 'a', newline='') as width_file:
			width_writer = csv.writer(width_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			width_writer.writerow((str(max_width), str(min_width)))

	def save_height(self, max_height, min_height, ipcl_type):
		output = self.data_output + ipcl_type
		with open(output + 'height.csv', 'a', newline='') as height_file:
			height_writer = csv.writer(height_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			height_writer.writerow((str(max_height), str(min_height)))

	def save_calibre(self, max_calibre, min_calibre, ipcl_type):
		output = self.data_output + ipcl_type
		with open(output + 'calibre.csv', 'a', newline='') as calibre_file:
			calibre_writer = csv.writer(calibre_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			calibre_writer.writerow((str(max_calibre), str(min_calibre)))

