# Filename: Diagnosis.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Created: 21/03/2019

from sklearn.svm import LinearSVC
import pandas as pd
from functools import reduce
from collections import Counter
import numpy as np


class Diagnosis:

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

		features_df = pd.DataFrame({
			'Width': width_list,
			'Height': height_list,
			'Red': red_list,
			'Green': green_list,
			'Blue': blue_list,
			'Length': length_list
		})

		print('\n', features_df)

	def svm(self, sample_amt):
		group1 = self.group1().sample(n=sample_amt)
		group2 = self.group2().sample(n=sample_amt)
		group3 = self.group3().sample(n=sample_amt)
		x = group1.append([group2, group3])
		print(x)
		y = self.create_y(sample_amt)

	def create_y(self, amt):
		y = [1 for i in range(amt)]
		y.extend([2 for i in range(amt)])
		y.extend([3 for i in range(amt)])
		y_df = pd.DataFrame({'Group': y})
		return y_df


	@staticmethod
	def group1():
		area = pd.DataFrame(pd.read_csv('../data_output/group1/area.csv'))
		colour = pd.DataFrame(pd.read_csv('../data_output/group1/colour.csv'))
		height = pd.DataFrame(pd.read_csv('../data_output/group1/height.csv'))
		length = pd.DataFrame(pd.read_csv('../data_output/group1/length.csv'))
		width = pd.DataFrame(pd.read_csv('../data_output/group1/width.csv'))

		area_subset = area.iloc[1:370000]
		colour_subset = colour.iloc[1:370000]
		height_subset = height.iloc[1:370000]
		length_subset = length.iloc[1:370000]
		width_subset = width.iloc[1:370000]

		ac = area_subset.join(colour_subset)
		ach = ac.join(height_subset)
		achl = ach.join(length_subset)
		group_1 = achl.join(width_subset)

		return group_1

	@staticmethod
	def group2():
		area = pd.DataFrame(pd.read_csv('../data_output/group2/area.csv'))
		colour = pd.DataFrame(pd.read_csv('../data_output/group2/colour.csv'))
		height = pd.DataFrame(pd.read_csv('../data_output/group2/height.csv'))
		length = pd.DataFrame(pd.read_csv('../data_output/group2/length.csv'))
		width = pd.DataFrame(pd.read_csv('../data_output/group2/width.csv'))

		area_subset = area.loc[1:1700000]
		colour_subset = colour.loc[1:1700000]
		height_subset = height.loc[1:1700000]
		length_subset = length.loc[1:1700000]
		width_subset = width.loc[1:1700000]

		ac = area_subset.join(colour_subset)
		ach = ac.join(height_subset)
		achl = ach.join(length_subset)
		group_2 = achl.join(width_subset)

		return group_2

	@staticmethod
	def group3():
		area = pd.DataFrame(pd.read_csv('../data_output/group3/area.csv'))
		colour = pd.DataFrame(pd.read_csv('../data_output/group3/colour.csv'))
		height = pd.DataFrame(pd.read_csv('../data_output/group3/height.csv'))
		length = pd.DataFrame(pd.read_csv('../data_output/group3/length.csv'))
		width = pd.DataFrame(pd.read_csv('../data_output/group3/width.csv'))

		area_subset = area.loc[1:3300000]
		colour_subset = colour.loc[1:3300000]
		height_subset = height.loc[1:3300000]
		length_subset = length.loc[1:3300000]
		width_subset = width.loc[1:3300000]

		ac = area_subset.join(colour_subset)
		ach = ac.join(height_subset)
		achl = ach.join(length_subset)
		group_3 = achl.join(width_subset)

		return group_3


	# @staticmethod
	# def calculate_means(lists):
	# 	means = list()
	# 	for elements in lists:
	# 		means.append(np.mean(elements))
	# 	return means
	#
	# @staticmethod
	# def calculate_medians(lists):
	# 	medians = list()
	# 	for elements in lists:
	# 		medians.append(np.median(elements))
	# 	return medians
	#
	# @staticmethod
	# def calculate_stds(lists):
	# 	stds = list()
	# 	for elements in lists:
	# 		stds.append(np.std(elements, dtype=np.float64))
	# 	return stds
	#
	# def calculate_modes(self, lists):
	# 	modes = list()
	# 	for elements in lists:
	# 		modes.append(self.get_mode(elements))
	# 	return modes
	#
	# @staticmethod
	# def get_mode(elements):
	# 	counter = Counter(elements)
	# 	if len(counter.most_common(1)) >= 1:
	# 		_, val = counter.most_common(1)[0]
	# 		return [x for x, y in counter.items() if y == val]
	# 	else:
	# 		return []
	#
	# def add_to_statistics(self, means, medians, stds, modes):
	# 	self.statistics.clear()
	# 	self.add_to_means(means)
	# 	self.add_to_medians(medians)
	# 	self.add_to_stds(stds)
	# 	self.add_to_modes(modes)
	#
	# def add_to_means(self, means):
	# 	self.statistics['Mean Width'] = means[0]
	# 	self.statistics['Mean Height'] = means[1]
	# 	self.statistics['Mean Rotation'] = means[2]
	# 	self.statistics['Mean Area'] = means[3]
	# 	self.statistics['Mean Colour'] = (means[4], means[5], means[6])
	# 	self.statistics['Mean Length'] = means[7]
	#
	# def add_to_medians(self, medians):
	# 	self.statistics['Median Width'] = medians[0]
	# 	self.statistics['Median Height'] = medians[1]
	# 	self.statistics['Median Rotation'] = medians[2]
	# 	self.statistics['Median Area'] = medians[3]
	# 	self.statistics['Median Colour'] = (medians[4], medians[5], medians[6])
	# 	self.statistics['Median Length'] = medians[7]
	#
	# def add_to_stds(self, stds):
	# 	self.statistics['StD Width'] = stds[0]
	# 	self.statistics['StD Height'] = stds[1]
	# 	self.statistics['StD Rotation'] = stds[2]
	# 	self.statistics['StD Area'] = stds[3]
	# 	self.statistics['StD Colour'] = (stds[4], stds[5], stds[6])
	# 	self.statistics['StD Length'] = stds[7]
	#
	# def add_to_modes(self, modes):
	# 	self.statistics['Mode Width'] = modes[0]
	# 	self.statistics['Mode Height'] = modes[1]
	# 	self.statistics['Mode Rotation'] = modes[2]
	# 	self.statistics['Mode Area'] = modes[3]
	# 	self.statistics['Mode Colour'] = (modes[4], modes[5], modes[6])
	# 	self.statistics['Mode Length'] = modes[7]
	#
	# def add_occurrences(self):
	# 	self.statistics['Mean Occurrences'] = np.mean(self.number_ipcls)
	# 	self.statistics['Median Occurrences'] = np.median(self.number_ipcls)
	# 	self.statistics['StD Occurrences'] = np.std(self.number_ipcls)
	# 	self.statistics['Mode Occurrences'] = self.get_mode(self.number_ipcls)
