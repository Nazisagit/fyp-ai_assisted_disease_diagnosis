# Filename: IPCLDiagnoser.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Opened 13/02/2019

import numpy as np
import math
from collections import Counter

# TODO
# 1. Explore idea of recalculating probability of diagnosing next IPCL type if the first type and next type
#    fall within a threshold
# 2.


class IPCLDiagnosis:

	def __init__(self, feature_tables, statistical_diagnoses_output):
		self.feature_tables = feature_tables

		self.statistical_diagnoses_output = statistical_diagnoses_output

		self.statistics = dict()

	def analyse_feature_tables(self):
		# Define vectors necessary for calculation
		rotation_vector = list()
		area_vector = list()
		length_vector = list()
		width_vector = list()
		height_vector = list()
		feature_vectors = list()

		# Uncomment to save statistical data
		# new_file = open(self.statistical_diagnoses_output + 'statistical_data.txt', 'w+')
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
			self.append_feature_vectors(feature_vectors, rotation_vector, area_vector, length_vector,
			                            width_vector, height_vector, calibre_vector)
			means = self.calculate_means(feature_vectors)
			medians = self.calculate_medians(feature_vectors)
			stds = self.calculate_stds(feature_vectors)
			modes = self.calculate_modes(feature_vectors)
			self.add_to_stats(means, medians, stds, modes)
			# Uncomment to print each table
			# self.print_to_console(means, medians, stds, modes, table_height)

			# Uncomment to save statistical data
			# self.save_statistical_data(means, medians, stds, modes, table_height, new_file)

		# Uncomment to save statistical data
		# new_file.close()

	@staticmethod
	def append_feature_vectors(features, rotation, area, length, width, height, calibre):
		features.append(rotation)
		features.append(area)
		features.append(length)
		features.append(width)
		features.append(height)
		features.append(calibre)

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
			stds.append(np.std(vector, dtype = np.float64))
		return stds

	def calculate_modes(self, vectors):
		modes = list()
		for vector in vectors:
			modes.append(self.get_mode(vector))
		return modes

	@staticmethod
	def calculate_calibres(area_vector, length_vector):
		calibre_vector = list()
		for area in area_vector:
			for length in length_vector:
				calibre_vector.append(area/length)
		return calibre_vector

	@staticmethod
	def get_mode(vector):
		counter = Counter(vector)
		if len(counter.most_common(1)) >= 1:
			_, val = counter.most_common(1)[0]
			return [x for x, y in counter.items() if y == val]
		else:
			return []

	def add_to_stats(self, means, medians, stds, modes):
		self.statistics.clear()
		# Record the results in the dictionary in the state
		self.statistics['Mean Rotation'] = means[0]
		self.statistics['Mean Area'] = means[1]
		self.statistics['Mean Length'] = means[2]
		self.statistics['Mean Width'] = means[3]
		self.statistics['Mean Height'] = means[4]
		self.statistics['Mean Calibre'] = means[5]

		self.statistics['Median Rotation'] = medians[0]
		self.statistics['Median Area'] = medians[1]
		self.statistics['Median Length'] = medians[2]
		self.statistics['Median Width'] = medians[3]
		self.statistics['Median Height'] = medians[4]
		self.statistics['Median Calibre'] = medians[5]

		self.statistics['StD Rotation'] = stds[0]
		self.statistics['StD Area'] = stds[1]
		self.statistics['StD Length'] = stds[2]
		self.statistics['StD Width'] = stds[3]
		self.statistics['StD Height'] = stds[4]
		self.statistics['StD Calibre'] = stds[5]

		self.statistics['Mode Rotation'] = modes[0]
		self.statistics['Mode Area'] = modes[1]
		self.statistics['Mode Length'] = modes[2]
		self.statistics['Mode Width'] = modes[3]
		self.statistics['Mode Height'] = modes[4]
		self.statistics['Mode Calibre'] = modes[5]

	@staticmethod
	def print_statistical_data(means, medians, stds, modes, table_height):
		print('----------------------------------------------------------------------------')
		print('Mean Rotation: ', means[0])
		print('Mean Area: ', means[1])
		print('Mean Length: ', means[2])
		print('Mean Width: ', means[3])
		print('Mean Height: ', means[4])
		print('Mean Calibre: ', means[5])
		print('----------------------------------------------------------------------------')
		print('Median Rotation: ', medians[0])
		print('Median Area: ', medians[1])
		print('Median Length: ', medians[2])
		print('Median Width: ', medians[3])
		print('Median Height: ', medians[4])
		print('Median Calibre: ', medians[5])
		print('----------------------------------------------------------------------------')
		print('StD Rotation: ', stds[0])
		print('StD Area: ', stds[1])
		print('StD Length: ', stds[2])
		print('StD Width: ', stds[3])
		print('StD Height: ', stds[4])
		print('StD Calibre: ', stds[5])
		print('----------------------------------------------------------------------------')
		print('Mode Rotation: ', modes[0])
		print('Mode Area: ', modes[1])
		print('Mode Length: ', modes[2])
		print('Mode Width: ', modes[3])
		print('Mode Height: ', modes[4])
		print('Mode Calibre: ', modes[5])
		print('----------------------------------------------------------------------------')
		print('Elements: ', table_height)
		print()

	@staticmethod
	def normalize(number):
		return (number * 100) - 100

	# Flattens the list. Used to diagnose mode features when multiple modes are present
	@staticmethod
	def flatten(values):
		# Define an output list
		output = []

		# Loop all values
		for value in values:
			if value is not None:
				if value[0] == True or value[0] == False:
					output.append(value)
				else:
					for element in value:
						output.append(element)

		return output

	@staticmethod
	def calculate_probability(data, mean, std):
		# Calculate probability of belonging to the class
		exponent = math.exp(-(math.pow(data - mean, 2) / (2 * math.pow(std, 2))))
		return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

	@staticmethod
	def save_statistical_data(means, medians, stds, modes, table_height, new_file):
		new_file.write('\n')
		new_file.write('----------------------------------------------------------------------------\n')
		new_file.write('Mean Rotation: ' + str(means[0]) + '\n')
		new_file.write('Mean Area: ' + str(means[1]) + '\n')
		new_file.write('Mean Length: ' + str(means[2]) + '\n')
		new_file.write('Mean Width: ' + str(means[3]) + '\n')
		new_file.write('Mean Height: ' + str(means[4]) + '\n')
		new_file.write('Mean Calibre: ' + str(means[5]) + '\n')
		new_file.write('----------------------------------------------------------------------------\n')
		new_file.write('Median Rotation: ' + str(medians[0]) + '\n')
		new_file.write('Median Area: ' + str(medians[1]) + '\n')
		new_file.write('Median Length: ' + str(medians[2]) + '\n')
		new_file.write('Median Width: ' + str(medians[3]) + '\n')
		new_file.write('Median Height: ' + str(medians[4]) + '\n')
		new_file.write('Median Calibre: ' + str(medians[5]) + '\n')
		new_file.write('----------------------------------------------------------------------------\n')
		new_file.write('StD Rotation: ' + str(stds[0]) + '\n')
		new_file.write('StD Area: ' + str(stds[1]) + '\n')
		new_file.write('StD Length: ' + str(stds[2]) + '\n')
		new_file.write('StD Width: ' + str(stds[3]) + '\n')
		new_file.write('StD Height: ' + str(stds[4]) + '\n')
		new_file.write('StD Calibre: ' + str(stds[5]) + '\n')
		new_file.write('----------------------------------------------------------------------------\n')
		new_file.write('Mode Rotation: ' + str(modes[0]) + '\n')
		new_file.write('Mode Area: ' + str(modes[1]) + '\n')
		new_file.write('Mode Length: ' + str(modes[2]) + '\n')
		new_file.write('Mode Width: ' + str(modes[3]) + '\n')
		new_file.write('Mode Height: ' + str(modes[4]) + '\n')
		new_file.write('Mode Calibre: ' + str(modes[5]) + '\n')
		new_file.write('----------------------------------------------------------------------------\n')
		new_file.write('Elements: ' + str(table_height))
		new_file.write('\n')

	def diagnose_by_type(self):
		# [Type 1, Type 2, Type 3, Type 4, Type 5]
		# Rotation
		mean_rotation = [1, 2, 3, 4, 5]
		std_rotation = [1, 2, 3, 4, 5]
		rotation = [mean_rotation, std_rotation]
		# Area
		mean_area = [1, 2, 3, 4, 5]
		std_area = [1, 2, 3, 4, 5]
		area = [mean_area, std_area]

		# Length
		mean_length = [1, 2, 3, 4, 5]
		std_length = [1, 2, 3, 4, 5]
		length = [mean_length, std_length]

		# Width
		mean_width = [1, 2, 3, 4, 5]
		std_width = [1, 2, 3, 4, 5]
		width = [mean_width, std_width]

		# Height
		mean_height = [1, 2, 3, 4, 5]
		std_height = [1, 2, 3, 4, 5]
		height = [mean_height, std_height]

		# Calibre
		mean_calibre = [1, 2, 3, 4, 5]
		std_calibre = [1, 2, 3, 4, 5]
		calibre = [mean_calibre, std_calibre]

		feature_measurements = [rotation, area, length, width, height, calibre]

		diagnoses = dict()
		diagnoses['Type 1'] = 1
		diagnoses['Type 2'] = 1
		diagnoses['Type 3'] = 1
		diagnoses['Type 4'] = 1
		diagnoses['Type 5'] = 1

		self.diagnose_type_1(feature_measurements, diagnoses)
		self.diagnose_type_2(feature_measurements, diagnoses)
		self.diagnose_type_3(feature_measurements, diagnoses)
		self.diagnose_type_4(feature_measurements, diagnoses)
		self.diagnose_type_5(feature_measurements, diagnoses)

		normalisation_constant = diagnoses['Type 1'] + diagnoses['Type 2']\
		                         + diagnoses['Type 3'] + diagnoses['Type 4'] + diagnoses['Type 5']

		self.print_results(diagnoses, normalisation_constant)

	def diagnose_type_1(self, feature_measurements, diagnoses):
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                  feature_measurements[0][0][0], feature_measurements[0][1][0])
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                  feature_measurements[1][0][0], feature_measurements[1][1][0])
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                  feature_measurements[2][0][0], feature_measurements[2][1][0])
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[3][0][0], feature_measurements[3][1][0])
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[4][0][0], feature_measurements[4][1][0])
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Calibre'],
		                                                  feature_measurements[5][0][0], feature_measurements[5][1][0])

	def diagnose_type_2(self, feature_measurements, diagnoses):
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                  feature_measurements[0][0][1], feature_measurements[0][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                  feature_measurements[1][0][1], feature_measurements[1][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                  feature_measurements[2][0][1], feature_measurements[2][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[3][0][1], feature_measurements[3][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[4][0][1], feature_measurements[4][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Calibre'],
		                                                  feature_measurements[5][0][1], feature_measurements[5][1][1])

	def diagnose_type_3(self, feature_measurements, diagnoses):
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                  feature_measurements[0][0][2], feature_measurements[0][1][2])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                  feature_measurements[1][0][2], feature_measurements[1][1][2])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                  feature_measurements[2][0][2], feature_measurements[2][1][2])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[3][0][2], feature_measurements[3][1][2])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[4][0][2], feature_measurements[4][1][2])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Calibre'],
		                                                  feature_measurements[5][0][2], feature_measurements[5][1][2])

	def diagnose_type_4(self, feature_measurements, diagnoses):
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                  feature_measurements[0][0][3], feature_measurements[0][1][3])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                  feature_measurements[1][0][3], feature_measurements[1][1][3])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                  feature_measurements[2][0][3], feature_measurements[2][1][3])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[3][0][3], feature_measurements[3][1][3])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[4][0][3], feature_measurements[4][1][3])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Calibre'],
		                                                  feature_measurements[5][0][3], feature_measurements[5][1][3])

	def diagnose_type_5(self, feature_measurements, diagnoses):
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                  feature_measurements[0][0][4], feature_measurements[0][1][4])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                  feature_measurements[1][0][4], feature_measurements[1][1][4])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                  feature_measurements[2][0][4], feature_measurements[2][1][4])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[3][0][4], feature_measurements[3][1][4])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[4][0][4], feature_measurements[4][1][4])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Calibre'],
		                                                  feature_measurements[5][0][4], feature_measurements[5][1][4])

	@staticmethod
	def print_results(diagnoses, normalisation_constant):
		if max(diagnoses.values()) == diagnoses['Type 1']:
			print('IPCL Type 1 identified. Probability: ', diagnoses['Type 1'] / normalisation_constant * 100)
		elif max(diagnoses.values()) == diagnoses['Type 2']:
			print('IPCL Type 2 identified. Probability: ', diagnoses['Type 2'] / normalisation_constant * 100)
		elif max(diagnoses.values()) == diagnoses['Type 3']:
			print('IPCL Type 3 identified. Probability: ', diagnoses['Type 3'] / normalisation_constant * 100)
		elif max(diagnoses.values()) == diagnoses['Type 4']:
			print('IPCL Type 4 identified. Probability: ', diagnoses['Type 4'] / normalisation_constant * 100)
		elif max(diagnoses.values()) == diagnoses['Type 5']:
			print('IPCL Type 5 identified. Probability: ', diagnoses['Type 5'] / normalisation_constant * 100)