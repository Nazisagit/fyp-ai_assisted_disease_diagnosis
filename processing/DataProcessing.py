# Filename: DataProcessing.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Opened 23/02/2019

import numpy as np
import math
from collections import Counter
import csv
import os


class DataProcessing:

	def __init__(self, feature_tables, data_output):
		self.feature_tables = feature_tables
		self.data_output = data_output

	def get_data(self):
		rotation_vector = list()
		area_vector = list()
		length_vector = list()
		width_vector = list()
		height_vector = list()
		for table in self.feature_tables:
			table_height = len(table)
			# For Rotation, Area, and Length
			for feature in [0, 1, 2, 3, 4]:
				# For all entries in the table
				for row in range(0, table_height):
					# Calculate Total Rotation
					if feature == 0:
						rotation_vector.append(table[row][feature])
					elif feature == 1:
						area_vector.append(table[row][feature])
					elif feature == 2:
						length_vector.append(table[row][feature])
					# Calculate Total Area
					elif feature == 3:
						width_vector.append(table[row][feature])
					# Calculate Total Length
					elif feature == 4:
						height_vector.append(table[row][feature])
			calibre_vector = self.calculate_calibres(area_vector, length_vector)
			max_min_rotation = self.get_max_min(rotation_vector)
			self.save_rotation(max_min_rotation, 'type1/')
			max_min_area = self.get_max_min(area_vector)
			self.save_area(max_min_area, 'type1/')
			max_min_length = self.get_max_min(length_vector)
			self.save_length(max_min_length, 'type1/')
			max_min_width = self.get_max_min(width_vector)
			self.save_width(max_min_width, 'type1/')
			max_min_height = self.get_max_min(height_vector)
			self.save_height(max_min_height, 'type1/')
			max_min_calibre = self.get_max_min(calibre_vector)
			self.save_calibre(max_min_calibre, 'type1/')

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

	@staticmethod
	def get_max_min(vector):
		return max(vector), min(vector)

	def save_rotation(self, max_min_rotation, ipcl_type):
		rotation = self.data_output + ipcl_type
		if not os.path.exists(rotation):
			os.makedirs(rotation)
		with open(rotation + 'rotation.csv', 'a', newline='') as rotation_file:
			rotation_writer = csv.writer(rotation_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			rotation_writer.writerow(max_min_rotation[0], max_min_rotation[1])

	def save_area(self, max_min_area, ipcl_type):
		area = self.data_output + ipcl_type
		if not os.path.exists(area):
			os.makedirs(area)
		with open(area + 'area.csv', 'a', newline='') as area_file:
			area_writer = csv.writer(area_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			area_writer.writerow(max_min_area[0], max_min_area[1])

	def save_length(self, max_min_length, ipcl_type):
		length = self.data_output + ipcl_type
		if not os.path.exists(length):
			os.makedirs(length)
		with open(length + 'length.csv', 'a', newline='') as length_file:
			length_writer = csv.writer(length_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			length_writer.writerow(max_min_length[0], max_min_length[1])

	def save_width(self, max_min_width, ipcl_type):
		width = self.data_output + ipcl_type
		if not os.path.exists(width):
			os.makedirs(width)
		with open(width + 'width.csv', 'a', newline='') as width_file:
			width_writer = csv.writer(width_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			width_writer.writerow(max_min_width[0], max_min_width[1])

	def save_height(self, max_min_height, ipcl_type):
		height = self.data_output + ipcl_type
		if not os.path.exists(height):
			os.makedirs(height)
		with open(height + 'height.csv', 'a', newline='') as height_file:
			height_writer = csv.writer(height_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			height_writer.writerow(max_min_height[0], max_min_height[1])

	def save_calibre(self, max_min_calibre, ipcl_type):
		calibre = self.data_output + ipcl_type
		if not os.path.exists(calibre):
			os.makedirs(calibre)
		with open(calibre + 'calibre.csv', 'a', newline='') as calibre_file:
			calibre_writer = csv.writer(calibre_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			calibre_writer.writerow(max_min_calibre[0], max_min_calibre[1])
