# Filename: IPCLDiagnoser.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Opened 13/02/2019

import numpy as np
import math
from collections import Counter


class IPCLDiagnoser:

	def __init__(self, feature_tables, statistical_diagnoses_output):
		self.feature_tables = feature_tables

		self.statistical_diagnoses_output = statistical_diagnoses_output

		self.statistics = dict()

	def analyse_feature_tables(self):
		# Define vectors necessary for calculation
		width_vector = list()
		height_vector = list()
		rotation_vector = list()
		area_vector = list()
		length_vector = list()
		feature_vectors = list()

		new_file = open(self.statistical_diagnoses_output + 'diagnostic_data.txt', 'a+')
		for table in self.feature_tables:
			table_height = len(table)
			# For Rotation, Area, and Length
			for feature in [1, 2, 3, 4, 5]:
				# For all entries in the table
				for row in range(0, table_height):
					# Calculate Total Rotation
					if feature == 1:
						width_vector.append(table[row][feature])
					elif feature == 2:
						height_vector.append(table[row][feature])
					elif feature == 3:
						rotation_vector.append(table[row][feature])
					# Calculate Total Area
					elif feature == 4:
						area_vector.append(table[row][feature])
					# Calculate Total Length
					elif feature == 5:
						length_vector.append(table[row][feature])

			calibre_vector = self.calculate_calibres(area_vector, length_vector)

			feature_vectors.append(rotation_vector)
			feature_vectors.append(width_vector)
			feature_vectors.append(height_vector)
			feature_vectors.append(area_vector)
			feature_vectors.append(length_vector)
			feature_vectors.append(calibre_vector)

			means = self.calculate_means(feature_vectors)
			medians = self.calculate_medians(feature_vectors)
			stds = self.calculate_stds(feature_vectors)
			modes = self.calculate_modes(feature_vectors)

			self.add_to_stats(means, medians, stds, modes)
			# self.print_to_console(means, medians, stds, modes, table_height)
			self.save_diagnostic_data(means, medians, stds, modes, table_height, new_file)

	def calculate_means(self, vectors):
		means = list()
		for vector in vectors:
			means.append(np.mean(vector))
		return means

	def calculate_medians(self, vectors):
		medians = list()
		for vector in vectors:
			medians.append(np.median(vector))
		return medians

	def calculate_stds(self, vectors):
		stds = list()
		for vector in vectors:
			stds.append(np.std(vector, dtype = np.float64))
		return stds

	def calculate_modes(self, vectors):
		modes = list()
		for vector in vectors:
			modes.append(self.get_mode(vector))
		return modes

	def get_mode(self, vector):
		counter = Counter(vector)
		if len(counter.most_common(1)) >= 1:
			_, val = counter.most_common(1)[0]
			return [x for x, y in counter.items() if y == val]
		else:
			return []

	def add_to_stats(self, means, medians, stds, modes):
		self.statistics.clear()
		# Record the results in the dictionary in the state
		self.statistics['Mean Rotation'] = means[1]
		self.statistics['Mean Area'] = means[2]
		self.statistics['Mean Length'] = means[3]
		self.statistics['Median Rotation'] = medians[1]
		self.statistics['Median Area'] = medians[2]
		self.statistics['Median Length'] = medians[3]
		self.statistics['SD Rotation'] = stds[1]
		self.statistics['SD Area'] = stds[2]
		self.statistics['SD Length'] = stds[3]
		self.statistics['Mode Rotation'] = modes[1]
		self.statistics['Mode Area'] = modes[2]
		self.statistics['Mode Length'] = modes[3]

	def print_to_console(self, means, medians, stds, modes, table_height):
		print('----------------------------------------------------------------------------')
		print('Mean Rotation', means[1])
		print('Mean Area', means[2])
		print('Mean Length', means[3])
		print('Mean Calibre', means[4])
		print('----------------------------------------------------------------------------')
		print('Median Rotation', medians[1])
		print('Median Area', medians[2])
		print('Median Length', medians[3])
		print('Median Calibre', medians[4])
		print('----------------------------------------------------------------------------')
		print('Std Rotation', stds[1])
		print('Std Area', stds[2])
		print('Std Length', stds[3])
		print('Std Calibre'), stds[4]
		print('----------------------------------------------------------------------------')
		print('Mode Rotation', modes[1])
		print('Mode Area', modes[2])
		print('Mode Length', modes[3])
		print('Mode Calibre', modes[4])
		print('----------------------------------------------------------------------------')
		print('Elements:', table_height)
		print()

	def normalize(self, number):
		return (number * 100) - 100

	# Flattens the list. Used to diagnose mode features when multiple modes are present
	def flatten(self, list):
		# Define an output list
		output = []

		# Loop all values
		for value in list:
			if value is not None:
				if value[0] == True or value[0] == False:
					output.append(value)
				else:
					for element in value:
						output.append(element)

		return output

	def calculate_probability(self, data, mean, std):
		# Calculate probability of belonging to the class
		exponent = math.exp(-(math.pow(data - mean, 2) / (2 * math.pow(std, 2))))
		return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

	def save_diagnostic_data(self, means, medians, stds, modes, table_height, new_file):
		for row in range(0, table_height):
			new_file.write('|  Means  |  Medians  |  Stds  | Modes  |')
			new_file.write(means[row] + '  |  ' + medians[row] + '  |  ' + stds[row] + '  |  ' + modes[row])
			new_file.write('\n')
		new_file.close()

	def calculate_calibres(self, area_vector, length_vector):
		calibre_vector = list()
		for row in range(len(area_vector)):
			calibre = area_vector[row]/length_vector[row]
			calibre_vector.append(calibre)
		return calibre_vector

	def diagnose(self):
		# Area
		mean_area = []
		std_area = []
		# Length
		mean_length = []



