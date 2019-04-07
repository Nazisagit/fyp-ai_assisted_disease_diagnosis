# Filename: component_analyzer.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Last modified: 06/04/2019

"""
Analyzes the features to determine the most important
features needed to diagnose the IPCLs.
"""

import pandas as pd
import matplotlib.pyplot as plt
from time import time

from sklearn.ensemble import ExtraTreesClassifier


def __load_files(group_folder):
	"""
	Loads the data from the csv feature files in one group folder.
	Loads them in chunksizes for speed.
	https://towardsdatascience.com/why-and-how-to-use-pandas-with-large-data-9594dda2ea4c
	:param group_folder: where the csv feature files of one group exist
	:return: pandas DataFrames of width, height, area, colour, length
	"""
	chunksize = 10 ** 6

	width_chunks = pd.read_csv(group_folder + 'width.csv', chunksize=chunksize)
	width_list = list()
	for chunk in width_chunks:
		width_list.append(chunk)

	height_chunks = pd.read_csv(group_folder + 'height.csv', chunksize=chunksize)
	height_list = list()
	for chunk in height_chunks:
		height_list.append(chunk)

	area_chunks = pd.read_csv(group_folder + 'area.csv', chunksize=chunksize)
	area_list = list()
	for chunk in area_chunks:
		area_list.append(chunk)

	colour_chunks = pd.read_csv(group_folder + 'colour.csv', chunksize=chunksize)
	colour_list = list()
	for chunk in colour_chunks:
		colour_list.append(chunk)

	length_chunks = pd.read_csv(group_folder + 'length.csv', chunksize=chunksize)
	length_list = list()
	for chunk in length_chunks:
		length_list.append(chunk)

	width = pd.concat(width_list)
	height = pd.concat(height_list)
	area = pd.concat(area_list)
	colour = pd.concat(colour_list)
	length = pd.concat(length_list)

	return width, height, area, colour, length


def __group(group_folder):
	"""
	Groups together the data from the csv feature files in one group folder.
	Loads them in chunksizes for speed.
	:param group_folder: where the csv feature files of one group exist
	:return: a pandas DataFrame of the grouped data
	"""

	width, height, area, colour, length = __load_files(group_folder)

	wh = width.join(height)
	wha = wh.join(area)
	whac = wha.join(colour)
	group = whac.join(length)

	return group


def __create_y(len_g1, len_g2, len_g3):
	"""
	Creates corresponding list of classes
	:param len_g1: length of group 1
	:param len_g2: length of group 2
	:param len_g3: length of group 3
	:return:
	"""
	y = [1 for i in range(len_g1)]
	y.extend([2 for i in range(len_g2)])
	y.extend([3 for i in range(len_g3)])
	return y


def __group_lengths(group_folder):
	"""
	Returns the lengths of each feature in the group
	:param group_folder: folder where the csv feature files should be
	:return: lengths of width, height, area, colour, length
	"""
	width, height, area, colour, length = __load_files(group_folder)

	print(len(width), len(height), len(area), len(colour), len(length))


def check_importance(directory):
	"""
	Performs feature importance and prints out the results to allow users
	to understand which features would be most useful for classification
	:param directory: folder where the group IPCL folders should be
	"""
	print('Checking feature importance')
	t0 = time()
	group1 = __group(directory + 'group1/')
	group2 = __group(directory + 'group2/')
	group3 = __group(directory + 'group3/')
	data = group1.append([group2, group3], ignore_index=True)
	y = __create_y(len(group1), len(group2), len(group3))
	etc = ExtraTreesClassifier(n_estimators=100)
	etc.fit(data, y)
	print('Feature importance complete in %0.3fs' % (time() - t0))
	print(etc.feature_importances_)
	feat_importances = pd.Series(etc.feature_importances_, index=data.columns)
	feat_importances.nlargest(10).plot(kind='barh')
	plt.show()


if __name__ == "__main__":
	# Prints out the amounts of a specific feature that has been collected
	# Provide complete path
	__group_lengths('D:/University/FYP/project/data_output/group1/')
	__group_lengths('D:/University/FYP/project/data_output/group2/')
	__group_lengths('D:/University/FYP/project/data_output/group3/')
	# Checks the importance of the features to decide what features are most important
	# Provide complete path
	check_importance('D:/University/FYP/project/data_output/')
