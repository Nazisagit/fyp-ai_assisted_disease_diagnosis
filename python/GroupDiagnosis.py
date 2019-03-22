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

	def analyse_feature_tables(self):
		width_list = list()
		height_list = list()
		area_list = list()
		red_list = list()
		blue_list = list()
		green_list = list()
		length_list = list()
		feature_lists = list()

		for table in self.feature_tables:
			table_height = len(table)
			for feature in [0, 1, 2, 3, 4]:
				for row in range(0, table_height):
					if feature == 0:
						width_list.append(table[row][feature])
					elif feature == 1:
						height_list.append(table[row][feature])
					elif feature == 2:
						area_list.append(table[row][feature])
					elif feature == 3:
						red_list.append(table[row][feature][0])
						blue_list.append(table[row][feature][1])
						green_list.append(table[row][feature][2])
					elif feature == 4:
						length_list.append(table[row][feature])

			self.append_feature_vectors(feature_lists, width_list, height_list,
			                            area_list, red_list, green_list, blue_list, length_list)
			means = self.calculate_means(feature_lists)
			medians = self.calculate_medians(feature_lists)
			stds = self.calculate_stds(feature_lists)
			modes = self.calculate_modes(feature_lists)
			self.add_to_statistics(means, medians, stds, modes)
			self.add_occurrences()
			self.diagnose_by_group()
		self.determine_group()

	@staticmethod
	def append_feature_vectors(features, width, height, area, red, green, blue, length):
		features.append(width)
		features.append(height)
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

	def diagnose_by_group(self):
		# [Group 1, Group 2, Group 3]
		# Width
		# 75% sample
		mean_width = [8.197891]
		std_width = [7.964949]
		# 240k samples
		# mean_width = [9.021655, 8.566543, 8.851637]
		# std_width = [15.120905, 15.000839, 14.102458]
		# 20% samples
		# mean_width = [8.799076, 8.85634, 8.961676]
		# std_width = [12.149808, 17.492623, 14.241984]
		width = [mean_width, std_width]

		# Height
		# 75% samples
		mean_height = [9.940834]
		std_height = [9.940834]
		# 240k samples
		# mean_height = [8.307154, 8.059593, 7.962122]
		# std_height = [9.51824, 7.330072, 8.534472]
		# height = [mean_height, std_height]
		# 20% samples
		# mean_height = [8.425792, 7.999119, 8.030263]
		# std_height = [8.739446, 7.641639, 8.386695]
		height = [mean_height, std_height]

		# Area
		# 75% samples
		mean_area = [27.976652]
		std_area = [69.162132]
		# 240k samples
		# mean_area = [21.028134, 21.106137, 24.210444]
		# std_area = [22.457414, 21.952237, 30.780212]
		# 20% samples
		# mean_area = [22.89487, 19.786446, 23.830924]
		# std_area = [24.03765, 20.068357, 30.412831]
		area = [mean_area, std_area]

		# Colour
		red = [(105.344278, 22.255299)]
		green = [(106.249847, 22.344215)]
		blue = [(84.181842, 19.954252)]
		# 240k samples
		# red = [(98.787334, 35.809122), (112.048579, 27.469161), (94.329682, 30.346502)]
		# green = [(94.672681, 31.230764), (102.974930, 27.163051), (94.329682, 30.346502)]
		# blue = [(74.228490, 26.914684), (81.262298, 23.969940), (73.678371, 27.756969)]
		# 20% samples
		# red = [(94.341010, 34.403288), (109.200473, 29.444711), (95.009487, 30.111865)]
		# green = [(92.774057, 30.284560), (103.166494, 27.837301), (95.042572, 32.266208)]
		# blue = [(72.472137, 26.133280), (81.543294, 24.635490), (74.605212, 28.221014)]

		# Length
		# 75% samples
		mean_length = [22.260356]
		std_length = [34.068365]
		# 240k samples
		# mean_length = [20.216815, 19.2365, 19.781513]
		# std_length = [21.653142, 20.086535, 21.430037]
		# length = [mean_length, std_length]
		# 20% samples
		# mean_length = [19.51346, 20.084864, 20.171453]
		# std_length = [19.165771, 22.686842, 22.417837]
		length = [mean_length, std_length]

		# Occurrences
		# 75% samples
		mean_occurrences = [140.05]
		std_occurrences = [79.76378]
		# 240k samples
		# mean_occurrences = [78.460714, 102.196429, 114.535714]
		# std_occurrences = [50.334778, 124.207137, 113.811115]
		# occurrences = [mean_occurrences, std_occurrences]
		# 20% samples
		# mean_occurrences = [76.568182, 117.62782, 93.274476]
		# std_occurrences = [50.400875, 127.334267, 83.574625]
		occurrences = [mean_occurrences, std_occurrences]

		feature_measurements = [width, height, area, red, green, blue, length, occurrences]
		self.init_diagnoses_group()
		self.diagnose_group_1(feature_measurements, self.diagnoses)
		self.diagnose_group_2(feature_measurements, self.diagnoses)
		self.diagnose_group_3(feature_measurements, self.diagnoses)

	def diagnose_group_1(self, feature_measurements, diagnoses):
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                   feature_measurements[0][0][0], feature_measurements[0][1][0])
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                   feature_measurements[1][0][0], feature_measurements[1][1][0])
		# diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Area'],
		#                                                    feature_measurements[2][0][0], feature_measurements[2][1][0])
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Colour'][0],
		                                                  feature_measurements[3][0][0], feature_measurements[3][0][1])
		# diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Colour'][1],
		#                                                   feature_measurements[4][0][0], feature_measurements[4][0][1])
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Colour'][2],
		                                                  feature_measurements[5][0][0], feature_measurements[5][0][1])
		# diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Length'],
		#                                                    feature_measurements[6][0][0], feature_measurements[6][1][0])
		diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Occurrences'],
		                                                  feature_measurements[7][0][0], feature_measurements[7][1][0])

	def diagnose_group_2(self, feature_measurements, diagnoses):
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                   feature_measurements[0][0][1], feature_measurements[0][1][1])
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                   feature_measurements[1][0][1], feature_measurements[1][1][1])
		# diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Area'],
		#                                                    feature_measurements[2][0][1], feature_measurements[2][1][1])
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Colour'][0],
		                                                  feature_measurements[3][1][0], feature_measurements[3][1][1])
		# diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Colour'][1],
		#                                                   feature_measurements[4][1][0], feature_measurements[4][1][1])
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Colour'][2],
		                                                  feature_measurements[5][1][0], feature_measurements[5][1][1])
		# diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Length'],
		#                                                    feature_measurements[6][0][1], feature_measurements[6][1][1])
		diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Occurrences'],
		                                                  feature_measurements[7][0][1], feature_measurements[7][1][1])

	def diagnose_group_3(self, feature_measurements, diagnoses):
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Width'],
		                                                   feature_measurements[0][0][2], feature_measurements[0][1][2])
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Height'],
		                                                   feature_measurements[1][0][2], feature_measurements[1][1][2])
		# diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Area'],
		#                                                    feature_measurements[2][0][2], feature_measurements[2][1][2])
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Colour'][0],
		                                                  feature_measurements[3][2][0], feature_measurements[3][2][1])
		# diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Colour'][1],
		#                                                   feature_measurements[4][2][0], feature_measurements[4][2][1])
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Colour'][2],
		                                                  feature_measurements[5][2][0], feature_measurements[5][2][1])
		# diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Length'],
		#                                                    feature_measurements[6][0][2], feature_measurements[6][1][2])
		diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Occurrences'],
		                                                   feature_measurements[7][0][2], feature_measurements[7][1][2])

	def normalisation_constant(self):
		return self.diagnoses['Group 1'] + self.diagnoses['Group 2'] + self.diagnoses['Group 3']

	def determine_group(self):
		if max(self.diagnoses.values()) == self.diagnoses['Group 1']:
			print('\nGroup 1 IPCL. Probability:', self.diagnoses['Group 1'] / self.normalisation_constant() * 100)
			print('\nGroup 2 IPCL. Probability:', self.diagnoses['Group 2'] / self.normalisation_constant() * 100)
			print('\nGroup 3 IPCL. Probability:', self.diagnoses['Group 3'] / self.normalisation_constant() * 100)
		elif max(self.diagnoses.values()) == self.diagnoses['Group 2']:
			print('\nGroup 2 IPCL. Probability:', self.diagnoses['Group 2'] / self.normalisation_constant() * 100)
			print('\nGroup 1 IPCL. Probability:', self.diagnoses['Group 1'] / self.normalisation_constant() * 100)
			print('\nGroup 3 IPCL. Probability:', self.diagnoses['Group 3'] / self.normalisation_constant() * 100)
		elif max(self.diagnoses.values()) == self.diagnoses['Group 3']:
			print('\nGroup 3 IPCL. Probability:', self.diagnoses['Group 3'] / self.normalisation_constant() * 100)
			print('\nGroup 2 IPCL. Probability:', self.diagnoses['Group 2'] / self.normalisation_constant() * 100)
			print('\nGroup 1 IPCL. Probability:', self.diagnoses['Group 1'] / self.normalisation_constant() * 100)
