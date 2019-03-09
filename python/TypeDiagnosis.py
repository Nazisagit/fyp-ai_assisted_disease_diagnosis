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
		self.votes = list()

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
			self.diagnose_by_type()
		self.determine_vote()

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
		mean_width = [9.346334, 9.078073, 9.132982, 8.928902, 9.368635]
		std_width = [18.73465, 12.9983, 20.3663, 19.162447, 15.812641]
		width = [mean_width, std_width]

		# Height
		mean_height = [8.348297, 8.353444, 8.573133, 8.156304, 8.348297]
		std_height = [11.19423, 9.646411, 14.358488, 9.35614, 11.19423]
		height = [mean_height, std_height]

		# Rotation
		mean_rotation = [41.63328, 40.947691, 40.304986, 41.538044, 41.63328]
		std_rotation = [25.998956, 27.419118, 27.317437, 26.001473, 25.998956]
		rotation = [mean_rotation, std_rotation]

		# Area
		mean_area = [18.240561, 24.356245, 29.304159, 20.630282, 25.405012]
		std_area = [20.637367, 28.316103, 69.969139, 24.624611, 37.663504]
		area = [mean_area, std_area]

		# Colour
		red = [(), (), (), (), ()]
		green = [(), (), (), (), ()]
		blue = [(), (), (), (), ()]

		# Length
		mean_length = [21.058968, 19.521215, 19.195485, 20.298138, 21.058968]
		std_length = [26.471581, 19.891479, 32.185072, 24.696921, 26.471581]
		length = [mean_length, std_length]

		feature_measurements = [width, height, rotation, area, red, green, blue, length]

		self.init_diagnoses_type()
		self.diagnose_type_1(feature_measurements, self.diagnoses)
		self.diagnose_type_2(feature_measurements, self.diagnoses)
		self.diagnose_type_3(feature_measurements, self.diagnoses)
		self.diagnose_type_4(feature_measurements, self.diagnoses)
		self.diagnose_type_5(feature_measurements, self.diagnoses)
		self.votes.append(self.vote_for_type(self.diagnoses))

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
		diagnoses['Type 1'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                  feature_measurements[2][0][0], feature_measurements[2][1][0])
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

	def diagnose_type_2(self, feature_measurements, diagnoses):
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[0][0][1], feature_measurements[0][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[1][0][1], feature_measurements[1][1][1])
		diagnoses['Type 2'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                  feature_measurements[2][0][1], feature_measurements[2][1][1])
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

	def diagnose_type_3(self, feature_measurements, diagnoses):
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[0][0][2], feature_measurements[0][1][2])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[1][0][2], feature_measurements[1][1][2])
		diagnoses['Type 3'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                  feature_measurements[2][0][2], feature_measurements[2][1][2])
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

	def diagnose_type_4(self, feature_measurements, diagnoses):
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[0][0][3], feature_measurements[0][1][3])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[1][0][3], feature_measurements[1][1][3])
		diagnoses['Type 4'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                  feature_measurements[2][0][3], feature_measurements[2][1][3])
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

	def diagnose_type_5(self, feature_measurements, diagnoses):
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                  feature_measurements[0][0][4], feature_measurements[0][1][4])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                  feature_measurements[1][0][4], feature_measurements[1][1][4])
		diagnoses['Type 5'] *= self.calculate_probability(self.statistics['Mean Rotation'],
		                                                  feature_measurements[2][0][4], feature_measurements[2][1][4])
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

	@staticmethod
	def vote_for_type(diagnoses):
		if max(diagnoses.values()) == diagnoses['Type 1']:
			return 1
		elif max(diagnoses.values()) == diagnoses['Type 2']:
			return 2
		elif max(diagnoses.values()) == diagnoses['Type 3']:
			return 3
		elif max(diagnoses.values()) == diagnoses['Type 4']:
			return 4
		elif max(diagnoses.values()) == diagnoses['Type 5']:
			return 5

	def normalisation_constant(self):
		return self.diagnoses['Type 1'] + self.diagnoses['Type 2'] + self.diagnoses['Type 3'] \
		       + self.diagnoses['Type 4'] + self.diagnoses['Type 5']

	def determine_vote(self):
		vote_1 = self.votes.count(1)
		vote_2 = self.votes.count(2)
		vote_3 = self.votes.count(3)
		vote_4 = self.votes.count(4)
		vote_5 = self.votes.count(5)
		if vote_1 > vote_2 and vote_1 > vote_3 and vote_1 > vote_4 and vote_1 > vote_5:
			print('\nIPCL Type 1 identified. Probability: ', self.diagnoses['Type 1'] / self.normalisation_constant() * 100)
		elif vote_2 > vote_1 and vote_2 > vote_3 and vote_2 > vote_4 and vote_2 > vote_5:
			print('\nIPCL Type 2 identified. Probability: ', self.diagnoses['Type 2'] / self.normalisation_constant() * 100)
		elif vote_3 > vote_1 and vote_3 > vote_2 and vote_3 > vote_4 and vote_3 > vote_5:
			print('\nIPCL Type 3 identified. Probability: ', self.diagnoses['Type 3'] / self.normalisation_constant() * 100)
		elif vote_4 > vote_1 and vote_4 > vote_2 and vote_4 > vote_3 and vote_4 > vote_5:
			print('\nIPCL Type 4 identified. Probability: ', self.diagnoses['Type 4'] / self.normalisation_constant() * 100)
		elif vote_5 > vote_1 and vote_5 > vote_2 and vote_5 > vote_3 and vote_5 > vote_4:
			print('\nIPCL Type 5 identified. Probability: ', self.diagnoses['Type 5'] / self.normalisation_constant() * 100)



