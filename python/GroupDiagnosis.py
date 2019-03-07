# Filename: GroupDiagnosis.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Opened 06/03/2019

import numpy as np
import math
from collections import Counter


class GroupDiagnosis:

	def __init__(self, feature_tables, statistical_diagnoses_output):
		self.feature_tables = feature_tables
		self.statistical_diagnoses_output = statistical_diagnoses_output
		self.diagnoses = dict()
		self.statistics = dict()
		self.votes = list()

	def analyse_feature_tables(self):
		rotation_list = list()
		area_list = list()
		length_list = list()
		width_list = list()
		height_list = list()
		feature_lists = list()
		for table in self.feature_tables:
			table_height = len(table)
			for feature in [0, 1, 2, 3, 4]:
				for row in range(0, table_height):
					if feature == 0:
						rotation_list.append(table[row][feature])
					elif feature == 1:
						area_list.append(table[row][feature])
					elif feature == 2:
						length_list.append(table[row][feature])
					elif feature == 3:
						width_list.append(table[row][feature])
					elif feature == 4:
						height_list.append(table[row][feature])
			self.append_feature_vectors(feature_lists, rotation_list, area_list, length_list, width_list, height_list)
			means = self.calculate_means(feature_lists)
			medians = self.calculate_medians(feature_lists)
			stds = self.calculate_stds(feature_lists)
			modes = self.calculate_modes(feature_lists)
			self.add_to_statistics(means, medians, stds, modes)
			self.diagnose_by_group()
		self.determine_vote()

	@staticmethod
	def append_feature_vectors(features, rotation, area, length, width, height):
		features.append(rotation)
		features.append(area)
		features.append(length)
		features.append(width)
		features.append(height)

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
		self.statistics['Mean Rotation'] = means[0]
		self.statistics['Mean Area'] = means[1]
		self.statistics['Mean Length'] = means[2]
		self.statistics['Mean Width'] = means[3]
		self.statistics['Mean Height'] = means[4]

	def add_to_medians(self, medians):
		self.statistics['Median Rotation'] = medians[0]
		self.statistics['Median Area'] = medians[1]
		self.statistics['Median Length'] = medians[2]
		self.statistics['Median Width'] = medians[3]
		self.statistics['Median Height'] = medians[4]

	def add_to_stds(self, stds):
		self.statistics['StD Rotation'] = stds[0]
		self.statistics['StD Area'] = stds[1]
		self.statistics['StD Length'] = stds[2]
		self.statistics['StD Width'] = stds[3]
		self.statistics['StD Height'] = stds[4]

	def add_to_modes(self, modes):
		self.statistics['Mode Rotation'] = modes[0]
		self.statistics['Mode Area'] = modes[1]
		self.statistics['Mode Length'] = modes[2]
		self.statistics['Mode Width'] = modes[3]
		self.statistics['Mode Height'] = modes[4]

	@staticmethod
	def calculate_probability(data, mean, std):
		# Calculate probability of belonging to the class
		exponent = math.exp(-(math.pow(data - mean, 2) / (2 * math.pow(std, 2))))
		return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

	def init_diagnoses_group(self):
		self.diagnoses['Group 1'] = 1
		self.diagnoses['Group 2'] = 1
		self.diagnoses['Group 3'] = 1

	def diagnose_by_group(self):
		# [Group 1, Group 2, Group 3]
		# Rotation
		mean_rotation = [41.28173191, 40.85833314, 41.63328]
		std_rotation = [26.974843, 27.317437, 25.998956]
		rotation = [mean_rotation, std_rotation]
		# Area
		mean_area = [23.27285058, 24.76160944, 25.405012]
		std_area = [27.295892, 69.969139, 37.663504]
		area = [mean_area, std_area]

		# Length
		mean_length = [19.96912873, 19.59377258, 21.058968]
		std_length = [21.022507, 32.185072, 26.471581]
		length = [mean_length, std_length]

		# Width
		mean_width = [9.127416917, 9.306886857, 9.368635]
		std_width = [14.228461, 20.3663, 15.812641]
		width = [mean_width, std_width]

		# Height
		mean_height = [8.416903906, 8.185942055, 8.348297]
		std_height = [10.243848, 14.358488, 11.19423]
		height = [mean_height, std_height]

		feature_measurements = [rotation, area, length, width, height]

		self.init_diagnoses_group()
		self.diagnose_group_1(feature_measurements, self.diagnoses)
		self.diagnose_group_2(feature_measurements, self.diagnoses)
		self.diagnose_group_3(feature_measurements, self.diagnoses)
		self.votes.append(self.vote_for_group(self.diagnoses))

	def diagnose_group_1(self, feature_measurements, diagnoses):
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                   feature_measurements[0][0][0], feature_measurements[0][1][0])
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                   feature_measurements[1][0][0], feature_measurements[1][1][0])
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                   feature_measurements[2][0][0], feature_measurements[2][1][0])
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                   feature_measurements[3][0][0], feature_measurements[3][1][0])
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                   feature_measurements[4][0][0], feature_measurements[4][1][0])

	def diagnose_group_2(self, feature_measurements, diagnoses):
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                   feature_measurements[0][0][1], feature_measurements[0][1][1])
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                   feature_measurements[1][0][1], feature_measurements[1][1][1])
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                   feature_measurements[2][0][1], feature_measurements[2][1][1])
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                   feature_measurements[3][0][1], feature_measurements[3][1][1])
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                   feature_measurements[4][0][1], feature_measurements[4][1][1])

	def diagnose_group_3(self, feature_measurements, diagnoses):
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                   feature_measurements[0][0][2], feature_measurements[0][1][2])
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Area'],
		                                                   feature_measurements[1][0][2], feature_measurements[1][1][2])
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Length'],
		                                                   feature_measurements[2][0][2], feature_measurements[2][1][2])
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                   feature_measurements[3][0][2], feature_measurements[3][1][2])
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                   feature_measurements[4][0][2], feature_measurements[4][1][2])

	@staticmethod
	def vote_for_group(diagnoses):
		if max(diagnoses.values()) == diagnoses['Group 1']:
			return 1
		elif max(diagnoses.values()) == diagnoses['Group 2']:
			return 2
		elif max(diagnoses.values()) == diagnoses['Group 3']:
			return 3

	def normalisation_constant(self):
		return self.diagnoses['Group 1'] \
		       + self.diagnoses['Group 2'] \
		       + self.diagnoses['Group 3']

	def determine_vote(self):
		vote_1 = self.votes.count(1)
		vote_2 = self.votes.count(2)
		vote_3 = self.votes.count(3)
		if vote_1 > vote_2 and vote_1 > vote_3:
			print('\nIPCL Group 1 identified. Probability: ', self.diagnoses['Group 1'] / self.normalisation_constant() * 100)
		elif vote_2 > vote_1 and vote_2 > vote_3:
			print('\nIPCL Group 2 identified. Probability: ', self.diagnoses['Group 2'] / self.normalisation_constant() * 100)
		elif vote_3 > vote_1 and vote_3 > vote_2:
			print('\nIPCL Group 3 identified. Probability: ', self.diagnoses['Group 3'] / self.normalisation_constant() * 100)
