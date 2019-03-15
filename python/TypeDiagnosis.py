# Filename: IPCLDiagnoser.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Opened 13/02/2019

import numpy as np
import math
from collections import Counter

# TODO
# 1. Implement voting system


class TypeDiagnosis:

	def __init__(self, feature_tables, number_ipcls, statistical_diagnoses_output):
		self.feature_tables = feature_tables
		self.number_ipcls = number_ipcls
		self.statistical_diagnoses_output = statistical_diagnoses_output
		self.diagnoses = dict()
		self.statistics = dict()
		self.one = 0
		self.two = 0
		self.three = 0
		self.four = 0
		self.five = 0
		self.votes = [self.one, self.two, self.three, self.four, self.five]

	def analyse_feature_tables(self):
		width_list = list()
		height_list = list()
		rotation_list = list()
		area_list = list()
		red_list = list()
		blue_list = list()
		green_list = list()
		length_list = list()
		feature_lists = list()

		for table in self.feature_tables:
			table_height = len(table)
			for feature in [0, 1, 2, 3, 4, 5]:
				for row in range(0, table_height):
					if feature == 0:
						width_list.append(table[row][feature])
					elif feature == 1:
						height_list.append(table[row][feature])
					elif feature == 2:
						rotation_list.append(table[row][feature])
					elif feature == 3:
						area_list.append(table[row][feature])
					elif feature == 4:
						red_list.append(table[row][feature][0])
						blue_list.append(table[row][feature][1])
						green_list.append(table[row][feature][2])
					elif feature == 5:
						length_list.append(table[row][feature])

			self.append_feature_vectors(feature_lists, width_list, height_list, rotation_list,
			                            area_list, red_list, green_list, blue_list, length_list)
			means = self.calculate_means(feature_lists)
			medians = self.calculate_medians(feature_lists)
			stds = self.calculate_stds(feature_lists)
			modes = self.calculate_modes(feature_lists)
			self.add_to_statistics(means, medians, stds, modes)
			self.add_occurrences()
			self.diagnose_by_type()
		# self.determine_vote()
		self.determine_type()

	@staticmethod
	def append_feature_vectors(features, width, height, rotation, area, red, green, blue, length):
		features.append(width)
		features.append(height)
		features.append(rotation)
		features.append(area)
		features.append(red)
		features.append(green)
		features.append(blue)
		features.append(length)

	@staticmethod
	def calculate_means(lists):
		means = list()
		for elements in lists:
			means.append(np.mean(elements))
		return means

	@staticmethod
	def calculate_medians(lists):
		medians = list()
		for elements in lists:
			medians.append(np.median(elements))
		return medians

	@staticmethod
	def calculate_stds(lists):
		stds = list()
		for elements in lists:
			stds.append(np.std(elements, dtype=np.float64))
		return stds

	def calculate_modes(self, lists):
		modes = list()
		for elements in lists:
			modes.append(self.get_mode(elements))
		return modes

	@staticmethod
	def get_mode(elements):
		counter = Counter(elements)
		if len(counter.most_common(1)) >= 1:
			_, val = counter.most_common(1)[0]
			return [x for x, y in counter.items() if y == val]
		else:
			return []

	def add_to_statistics(self, means, medians, stds, modes):
		self.statistics.clear()
		self.add_to_means(means)
		self.add_to_medians(medians)
		self.add_to_stds(stds)
		self.add_to_modes(modes)

	def add_to_means(self, means):
		self.statistics['Mean Width'] = means[0]
		self.statistics['Mean Height'] = means[1]
		self.statistics['Mean Rotation'] = means[2]
		self.statistics['Mean Area'] = means[3]
		self.statistics['Mean Colour'] = (means[4], means[5], means[6])
		self.statistics['Mean Length'] = means[7]

	def add_to_medians(self, medians):
		self.statistics['Median Width'] = medians[0]
		self.statistics['Median Height'] = medians[1]
		self.statistics['Median Rotation'] = medians[2]
		self.statistics['Median Area'] = medians[3]
		self.statistics['Median Colour'] = (medians[4], medians[5], medians[6])
		self.statistics['Median Length'] = medians[7]

	def add_to_stds(self, stds):
		self.statistics['StD Width'] = stds[0]
		self.statistics['StD Height'] = stds[1]
		self.statistics['StD Rotation'] = stds[2]
		self.statistics['StD Area'] = stds[3]
		self.statistics['StD Colour'] = (stds[4], stds[5], stds[6])
		self.statistics['StD Length'] = stds[7]

	def add_to_modes(self, modes):
		self.statistics['Mode Width'] = modes[0]
		self.statistics['Mode Height'] = modes[1]
		self.statistics['Mode Rotation'] = modes[2]
		self.statistics['Mode Area'] = modes[3]
		self.statistics['Mode Colour'] = (modes[4], modes[5], modes[6])
		self.statistics['Mode Length'] = modes[7]

	def add_occurrences(self):
		self.statistics['Mean Occurrences'] = np.mean(self.number_ipcls)
		self.statistics['Median Occurrences'] = np.median(self.number_ipcls)
		self.statistics['StD Occurrences'] = np.std(self.number_ipcls)
		self.statistics['Mode Occurrences'] = self.get_mode(self.number_ipcls)

	@staticmethod
	def calculate_probability(data, mean, std):
		# Calculate probability of belonging to the class
		exponent = math.exp(-(math.pow(data - mean, 2) / (2 * math.pow(std, 2))))
		return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

	def diagnose_by_type(self):
		# [Type 1, Type 2, Type 3, Type 4, Type 5]
		# Width
		mean_width = [9.150127, 8.635681, 8.262123, 8.880439, 9.003234]
		std_width = [18.114578, 10.766246, 11.912546, 17.903905, 14.529328]
		width = [mean_width, std_width]

		# Height
		mean_height = [8.171219, 8.302546, 8.113451, 8.056456, 8.000869]
		std_height = [10.439326, 8.089224, 7.159418, 7.469403, 8.382132]
		height = [mean_height, std_height]

		# Rotation
		mean_rotation = [90.0, 87.078689, 90.0, 89.478681, 87.755775]
		std_rotation = [0.01, 11.09079, 0.01, 4.815404, 9.795837]
		rotation = [mean_rotation, std_rotation]

		# Area
		mean_area = [18.033632, 24.1761, 21.788223, 20.445338, 24.093709]
		std_area = [19.213505, 27.009009, 20.448601, 22.584755, 30.801586]
		area = [mean_area, std_area]

		# Colour
		red = [(109.973664, 33.833426), (86.156104, 33.698000),
		       (111.147070, 26.217299), (111.496416, 28.218792),
		       (96.131059, 30.058415)]
		green = [(99.486546, 31.702755), (89.165103, 29.457098),
		         (100.108983, 26.862058), (104.924801, 27.511826),
		         (95.463959, 32.361538)]
		blue = [(78.433364, 27.482859), (69.321456, 25.324556),
		        (78.627009, 23.472018), (83.179267, 24.462849),
		        (74.991972, 28.361276)]

		# Length
		mean_length = [20.940169, 18.949608, 18.463331, 20.057924, 20.098817]
		std_length = [23.902838, 17.637065, 16.979052, 22.710487, 22.320908]
		length = [mean_length, std_length]

		mean_occurrences = [68.55, 81.356667, 52.086957, 148.2185, 102.851537]
		std_occurrences = [44.753413, 52.718183, 51.079398, 146.474588, 104.646254]
		occurrences = [mean_occurrences, std_occurrences]

		feature_measurements = [width, height, rotation, area, red, green, blue, length, occurrences]

		self.init_diagnoses_type()
		self.diagnose_type_1(feature_measurements, self.diagnoses)
		self.diagnose_type_2(feature_measurements, self.diagnoses)
		self.diagnose_type_3(feature_measurements, self.diagnoses)
		self.diagnose_type_4(feature_measurements, self.diagnoses)
		self.diagnose_type_5(feature_measurements, self.diagnoses)
		# self.vote_for_type(self.diagnoses)

	def init_diagnoses_type(self):
		self.diagnoses['Type 1'] = 1
		self.diagnoses['Type 2'] = 1
		self.diagnoses['Type 3'] = 1
		self.diagnoses['Type 4'] = 1
		self.diagnoses['Type 5'] = 1

	def diagnose_type_1(self, feature_measurements, diagnoses):
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[0][0][0], feature_measurements[0][1][0])
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[1][0][0], feature_measurements[1][1][0])
		# diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		#                                                   feature_measurements[2][0][0], feature_measurements[2][1][0])
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                  feature_measurements[3][0][0], feature_measurements[3][1][0])
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Colour'][0],
		                                                  feature_measurements[4][0][0], feature_measurements[4][0][1])
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Colour'][1],
		                                                  feature_measurements[5][0][0], feature_measurements[5][0][1])
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Colour'][2],
		                                                  feature_measurements[6][0][0], feature_measurements[6][0][1])
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                  feature_measurements[7][0][0], feature_measurements[7][1][0])
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Occurrences'],
		                                                  feature_measurements[8][0][0], feature_measurements[8][1][0])

	def diagnose_type_2(self, feature_measurements, diagnoses):
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[0][0][1], feature_measurements[0][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[1][0][1], feature_measurements[1][1][1])
		# diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		#                                                   feature_measurements[2][0][1], feature_measurements[2][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                  feature_measurements[3][0][1], feature_measurements[3][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Colour'][0],
		                                                  feature_measurements[4][1][0], feature_measurements[4][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Colour'][1],
		                                                  feature_measurements[5][1][0], feature_measurements[5][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Colour'][2],
		                                                  feature_measurements[6][1][0], feature_measurements[6][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                  feature_measurements[7][0][1], feature_measurements[7][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Occurrences'],
		                                                  feature_measurements[8][0][1], feature_measurements[8][1][1])

	def diagnose_type_3(self, feature_measurements, diagnoses):
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[0][0][2], feature_measurements[0][1][2])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[1][0][2], feature_measurements[1][1][2])
		# diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		#                                                   feature_measurements[2][0][2], feature_measurements[2][1][2])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                  feature_measurements[3][0][2], feature_measurements[3][1][2])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Colour'][0],
		                                                  feature_measurements[4][2][0], feature_measurements[4][2][1])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Colour'][1],
		                                                  feature_measurements[5][2][0], feature_measurements[5][2][1])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Colour'][2],
		                                                  feature_measurements[6][2][0], feature_measurements[6][2][1])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                  feature_measurements[7][0][2], feature_measurements[7][1][2])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Occurrences'],
		                                                  feature_measurements[8][0][2], feature_measurements[8][1][2])

	def diagnose_type_4(self, feature_measurements, diagnoses):
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[0][0][3], feature_measurements[0][1][3])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[1][0][3], feature_measurements[1][1][3])
		# diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		#                                                   feature_measurements[2][0][3], feature_measurements[2][1][3])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                  feature_measurements[3][0][3], feature_measurements[3][1][3])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Colour'][0],
		                                                  feature_measurements[4][3][0], feature_measurements[4][3][1])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Colour'][1],
		                                                  feature_measurements[5][3][0], feature_measurements[5][3][1])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Colour'][2],
		                                                  feature_measurements[6][3][0], feature_measurements[6][3][1])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                  feature_measurements[7][0][3], feature_measurements[7][1][3])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Occurrences'],
		                                                  feature_measurements[8][0][3], feature_measurements[8][1][3])

	def diagnose_type_5(self, feature_measurements, diagnoses):
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[0][0][4], feature_measurements[0][1][4])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[1][0][4], feature_measurements[1][1][4])
		# diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		#                                                   feature_measurements[2][0][4], feature_measurements[2][1][4])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                  feature_measurements[3][0][4], feature_measurements[3][1][4])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Colour'][0],
		                                                  feature_measurements[4][4][0], feature_measurements[4][4][1])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Colour'][1],
		                                                  feature_measurements[5][4][0], feature_measurements[5][4][1])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Colour'][2],
		                                                  feature_measurements[6][4][0], feature_measurements[6][4][1])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                  feature_measurements[7][0][4], feature_measurements[7][1][4])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Occurrences'],
		                                                  feature_measurements[8][0][4], feature_measurements[8][1][4])

	# def vote_for_type(self, diagnoses):
	# 	if max(diagnoses.values()) == diagnoses['Type 1']:
	# 		self.one += 1
	# 	elif max(diagnoses.values()) == diagnoses['Type 2']:
	# 		self.two += 1
	# 	elif max(diagnoses.values()) == diagnoses['Type 3']:
	# 		self.three += 1
	# 	elif max(diagnoses.values()) == diagnoses['Type 4']:
	# 		self.four += 1
	# 	elif max(diagnoses.values()) == diagnoses['Type 5']:
	# 		self.five += 1

	def normalisation_constant(self):
		return self.diagnoses['Type 1'] + self.diagnoses['Type 2'] + self.diagnoses['Type 3'] \
		       + self.diagnoses['Type 4'] + self.diagnoses['Type 5']

	# def determine_vote(self):
	# 	if np.argmax(self.votes) == 0:
	# 		print('\nType 1 IPCL. Probability:', self.diagnoses['Type 1'] / self.normalisation_constant() * 100)
	# 	elif np.argmax(self.votes) == 1:
	# 		print('\nType 2 IPCL. Probability:', self.diagnoses['Type 2'] / self.normalisation_constant() * 100)
	# 	elif np.argmax(self.votes) == 2:
	# 		print('\nType 3 IPCL. Probability:', self.diagnoses['Type 3'] / self.normalisation_constant() * 100)
	# 	elif np.argmax(self.votes) == 3:
	# 		print('\nType 4 IPCL. Probability:', self.diagnoses['Type 4'] / self.normalisation_constant() * 100)
	# 	elif np.argmax(self.votes) == 4:
	# 		print('\nType 5 IPCL. Probability:', self.diagnoses['Type 5'] / self.normalisation_constant() * 100)#

	def determine_type(self):
		if max(self.diagnoses.values()) == self.diagnoses['Type 1']:
			print('\nType 1 IPCL. Probability:', self.diagnoses['Type 1'] / self.normalisation_constant() * 100)
		elif max(self.diagnoses.values()) == self.diagnoses['Type 2']:
			print('\nType 2 IPCL. Probability:', self.diagnoses['Type 2'] / self.normalisation_constant() * 100)
		elif max(self.diagnoses.values()) == self.diagnoses['Type 3']:
			print('\nType 3 IPCL. Probability:', self.diagnoses['Type 3'] / self.normalisation_constant() * 100)
		elif max(self.diagnoses.values()) == self.diagnoses['Type 4']:
			print('\nType 4 IPCL. Probability:', self.diagnoses['Type 4'] / self.normalisation_constant() * 100)
		elif max(self.diagnoses.values()) == self.diagnoses['Type 5']:
			print('\nType 5 IPCL. Probability:', self.diagnoses['Type 5'] / self.normalisation_constant() * 100)
