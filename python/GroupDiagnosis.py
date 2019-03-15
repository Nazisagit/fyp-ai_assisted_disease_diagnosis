# Filename: GroupDiagnosis.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Opened 06/03/2019

import numpy as np
import math
from collections import Counter


class GroupDiagnosis:

	def __init__(self, feature_tables, number_ipcls, statistical_diagnoses_output):
		self.feature_tables = feature_tables
		self.number_ipcls = number_ipcls
		self.statistical_diagnoses_output = statistical_diagnoses_output
		self.diagnoses = dict()
		self.statistics = dict()
		self.votes = [0, 0, 0]

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
			self.diagnose_by_group()
		# self.determine_vote()
		self.determine_group()

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

	def init_diagnoses_group(self):
		self.diagnoses['Group 1'] = 1
		self.diagnoses['Group 2'] = 1
		self.diagnoses['Group 3'] = 1

	# def diagnose_by_group(self):
	# 	# [Group 1, Group 2, Group 3]
	# 	# Width
	# 	mean_width = [9.150127, 8.262123, 9.003234]
	# 	std_width = [18.114578, 11.912546, 14.529328]
	# 	width = [mean_width, std_width]
	#
	# 	# Height
	# 	mean_height = [8.171219, 8.113451, 8.000869]
	# 	std_height = [10.439326, 7.159418, 8.382132]
	# 	height = [mean_height, std_height]
	#
	# 	# Rotation
	# 	mean_rotation = [90.0, 90.0, 87.755775]
	# 	std_rotation = [0.01, 0.01, 9.795837]
	# 	rotation = [mean_rotation, std_rotation]
	#
	# 	# Area
	# 	mean_area = [18.033632, 21.788223, 24.093709]
	# 	std_area = [19.213505, 20.448601, 30.801586]
	# 	area = [mean_area, std_area]
	#
	# 	# Colour
	# 	red = [(109.973664, 33.833426), (111.147070, 26.217299), (96.131059, 30.058415)]
	# 	green = [(99.486546, 31.702755), (100.108983, 26.862058), (95.463959, 32.361538)]
	# 	blue = [(78.433364, 27.482859), (78.627009, 23.472018), (74.991972, 28.361276)]
	#
	# 	# Length
	# 	mean_length = [20.940169, 18.463331, 20.098817]
	# 	std_length = [23.902838, 16.979052, 22.320908]
	# 	length = [mean_length, std_length]
	#
	# 	# Occurrences
	# 	mean_occurrences = [68.55, 52.086957, 102.851537]
	# 	std_occurrences = [44.753413, 51.079398, 104.646254]
	# 	occurrences = [mean_occurrences, std_occurrences]
	#
	# 	feature_measurements = [width, height, rotation, area, red, green, blue, length, occurrences]
	# 	self.init_diagnoses_group()
	# 	self.diagnose_group_1(feature_measurements, self.diagnoses)
	# 	self.diagnose_group_2(feature_measurements, self.diagnoses)
	# 	self.diagnose_group_3(feature_measurements, self.diagnoses)
		# self.vote_for_group()

	def diagnose_by_group(self):
		# [Group 1, Group 2, Group 3]
		# Width
		mean_width = [9.150127, 8.262123, 9.003234]
		std_width = [18.114578, 11.912546, 14.529328]
		width = [mean_width, std_width]

		# Height
		mean_height = [8.171219, 8.113451, 8.000869]
		std_height = [10.439326, 7.159418, 8.382132]
		height = [mean_height, std_height]

		# Rotation
		mean_rotation = [90.0, 90.0, 87.755775]
		std_rotation = [0.01, 0.01, 9.795837]
		rotation = [mean_rotation, std_rotation]

		# Area
		mean_area = [18.033632, 21.788223, 24.093709]
		std_area = [19.213505, 20.448601, 30.801586]
		area = [mean_area, std_area]

		# Colour
		red = [(109.973664, 33.833426), (111.147070, 26.217299), (96.131059, 30.058415)]
		green = [(99.486546, 31.702755), (100.108983, 26.862058), (95.463959, 32.361538)]
		blue = [(78.433364, 27.482859), (78.627009, 23.472018), (74.991972, 28.361276)]

		# Length
		mean_length = [20.940169, 18.463331, 20.098817]
		std_length = [23.902838, 16.979052, 22.320908]
		length = [mean_length, std_length]

		# Occurrences
		mean_occurrences = [68.55, 52.086957, 102.851537]
		std_occurrences = [44.753413, 51.079398, 104.646254]
		occurrences = [mean_occurrences, std_occurrences]

		feature_measurements = [width, height, rotation, area, red, green, blue, length, occurrences]
		self.init_diagnoses_group()
		self.diagnose_group_1(feature_measurements, self.diagnoses)
		self.diagnose_group_2(feature_measurements, self.diagnoses)
		self.diagnose_group_3(feature_measurements, self.diagnoses)

	def diagnose_group_1(self, feature_measurements, diagnoses):
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                   feature_measurements[0][0][0], feature_measurements[0][1][0])
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                   feature_measurements[1][0][0], feature_measurements[1][1][0])
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                   feature_measurements[2][0][0], feature_measurements[2][1][0])
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                   feature_measurements[3][0][0], feature_measurements[3][1][0])
		# diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Colour'][0],
		#                                                   feature_measurements[4][0][0], feature_measurements[4][0][1])
		# diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Colour'][1],
		#                                                   feature_measurements[5][0][0], feature_measurements[5][0][1])
		# diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Colour'][2],
		#                                                   feature_measurements[6][0][0], feature_measurements[6][0][1])
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                   feature_measurements[7][0][0], feature_measurements[7][1][0])
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Occurrences'],
		                                                  feature_measurements[8][0][0], feature_measurements[8][1][0])

	def diagnose_group_2(self, feature_measurements, diagnoses):
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                   feature_measurements[0][0][1], feature_measurements[0][1][1])
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                   feature_measurements[1][0][1], feature_measurements[1][1][1])
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                   feature_measurements[2][0][1], feature_measurements[2][1][1])
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                   feature_measurements[3][0][1], feature_measurements[3][1][1])
		# diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Colour'][0],
		#                                                   feature_measurements[4][1][0], feature_measurements[4][1][1])
		# diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Colour'][1],
		#                                                   feature_measurements[5][1][0], feature_measurements[5][1][1])
		# diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Colour'][2],
		#                                                   feature_measurements[6][1][0], feature_measurements[6][1][1])
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                   feature_measurements[7][0][1], feature_measurements[7][1][1])
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Occurrences'],
		                                                  feature_measurements[8][0][1], feature_measurements[8][1][1])

	def diagnose_group_3(self, feature_measurements, diagnoses):
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                   feature_measurements[0][0][2], feature_measurements[0][1][2])
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                   feature_measurements[1][0][2], feature_measurements[1][1][2])
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                   feature_measurements[2][0][2], feature_measurements[2][1][2])
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                   feature_measurements[3][0][2], feature_measurements[3][1][2])
		# diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Colour'][0],
		#                                                   feature_measurements[4][2][0], feature_measurements[4][2][1])
		# diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Colour'][1],
		#                                                   feature_measurements[5][2][0], feature_measurements[5][2][1])
		# diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Colour'][2],
		#                                                   feature_measurements[6][2][0], feature_measurements[6][2][1])
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                   feature_measurements[7][0][2], feature_measurements[7][1][2])
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Occurrences'],
		                                                   feature_measurements[8][0][2], feature_measurements[8][1][2])

	def vote_for_group(self):
		max_key = max(self.diagnoses, key=lambda k: self.diagnoses[k])
		if max_key == 'Group 1':
			self.votes[0] += 1
		elif max_key == 'Group 1':
			self.votes[1] += 1
		elif max_key == 'Group 3':
			self.votes[2] += 1

	def normalisation_constant(self):
		return self.diagnoses['Group 1'] + self.diagnoses['Group 2'] + self.diagnoses['Group 3']

	def determine_vote(self):
		if self.votes.index(max(self.votes)) == 0:
			print('\nGroup 1 IPCL. Probability:', self.diagnoses['Group 1'] / self.normalisation_constant() * 100)
			print(self.votes)
		elif self.votes.index(max(self.votes)) == 1:
			print('\nGroup 2 IPCL. Probability:', self.diagnoses['Group 2'] / self.normalisation_constant() * 100)
			print(self.votes)
		elif self.votes.index(max(self.votes)) == 2:
			print('\nGroup 3 IPCL. Probability:', self.diagnoses['Group 3'] / self.normalisation_constant() * 100)
			print(self.votes)

	def determine_group(self):
		if max(self.diagnoses.values()) == self.diagnoses['Group 1']:
			print('\nGroup 1 IPCL. Probability:', self.diagnoses['Group 1'] / self.normalisation_constant() * 100)
		elif max(self.diagnoses.values()) == self.diagnoses['Group 2']:
			print('\nGroup 2 IPCL. Probability:', self.diagnoses['Group 2'] / self.normalisation_constant() * 100)
		elif max(self.diagnoses.values()) == self.diagnoses['Group 3']:
			print('\nGroup 3 IPCL. Probability:', self.diagnoses['Group 3'] / self.normalisation_constant() * 100)


