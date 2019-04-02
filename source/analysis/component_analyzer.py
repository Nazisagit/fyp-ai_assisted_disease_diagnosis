# Filename: component_analyzer.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Created: 21/03/2019


import pandas as pd
import dask.dataframe as dd

import matplotlib.pyplot as plt
from time import time

from sklearn.ensemble import ExtraTreesClassifier


def __group(directory):
	chunksize = 10 ** 5

	width_chunks = pd.read_csv(directory + 'width.csv', chunksize=chunksize)
	width_list = list()
	for chunk in width_chunks:
		width_list.append(chunk)

	height_chunks = pd.read_csv(directory + 'height.csv', chunksize=chunksize)
	height_list = list()
	for chunk in height_chunks:
		height_list.append(chunk)

	area_chunks = pd.read_csv(directory + 'area.csv', chunksize=chunksize)
	area_list = list()
	for chunk in area_chunks:
		area_list.append(chunk)

	colour_chunks = pd.read_csv(directory + 'colour.csv', chunksize=chunksize)
	colour_list = list()
	for chunk in colour_chunks:
		colour_list.append(chunk)

	length_chunks = pd.read_csv(directory + 'length.csv', chunksize=chunksize)
	length_list = list()
	for chunk in length_chunks:
		length_list.append(chunk)

	width = pd.concat(width_list)
	height = pd.concat(height_list)
	area = pd.concat(area_list)
	colour = pd.concat(colour_list)
	length = pd.concat(length_list)

	wh = width.join(height)
	wha = wh.join(area)
	whac = wha.join(colour)
	group = whac.join(length)

	return group


def __large_group(directory, frac):
	width = dd.read_csv(directory + 'width.csv')
	height = dd.read_csv(directory + 'height.csv')
	area = dd.read_csv(directory + 'area.csv')
	colour = dd.read_csv(directory + 'colour.csv')
	length = dd.read_csv(directory + 'length.csv')

	wh = width.sample(frac=frac).join(height.sample(frac=frac))
	wha = wh.join(area.sample(frac=frac))
	whac = wha.join(colour.sample(frac=frac))
	group = whac.join(length.sample(frac=frac))

	return group


def __create_y(len_g1, len_g2, len_g3):
	y = [1 for i in range(len_g1)]
	y.extend([2 for i in range(len_g2)])
	y.extend([3 for i in range(len_g3)])
	return y


def __group_lengths(directory):
	chunksize = 10 ** 7

	width_chunks = pd.read_csv(directory + 'width.csv', chunksize=chunksize)
	width_list = list()
	for chunk in width_chunks:
		width_list.append(chunk)

	height_chunks = pd.read_csv(directory + 'height.csv', chunksize=chunksize)
	height_list = list()
	for chunk in height_chunks:
		height_list.append(chunk)

	area_chunks = pd.read_csv(directory + 'area.csv', chunksize=chunksize)
	area_list = list()
	for chunk in area_chunks:
		area_list.append(chunk)

	colour_chunks = pd.read_csv(directory + 'colour.csv', chunksize=chunksize)
	colour_list = list()
	for chunk in colour_chunks:
		colour_list.append(chunk)

	length_chunks = pd.read_csv(directory + 'length.csv', chunksize=chunksize)
	length_list = list()
	for chunk in length_chunks:
		length_list.append(chunk)

	width = pd.concat(width_list)
	height = pd.concat(height_list)
	area = pd.concat(area_list)
	colour = pd.concat(colour_list)
	length = pd.concat(length_list)

	return [len(width), len(height), len(area), len(colour), len(length)]


def __large_group_lengths(directory, frac):
	width = dd.read_csv(directory + 'width.csv')
	height = dd.read_csv(directory + 'height.csv')
	area = dd.read_csv(directory + 'area.csv')
	colour = dd.read_csv(directory + 'colour.csv')
	length = dd.read_csv(directory + 'length.csv')

	return [len(width.sample(frac=frac).index), len(height.sample(frac=frac).index), len(area.sample(frac=frac).index),
	        len(colour.sample(frac=frac).index), len(length.sample(frac=frac).index)]


def __check_importance(directory):
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


def __large_check_importance(directory, frac):
	print('Checking feature importance for large files')
	t0 = time()
	group1 = __large_group(directory + 'group1/', frac)
	group2 = __large_group(directory + 'group2/', frac)
	group3 = __large_group(directory + 'group3/', frac)
	grouped = group1.append(group2)
	data = grouped.append(group3)
	length_group1 = max(__large_group_lengths(directory + 'group1/', frac))
	length_group2 = max(__large_group_lengths(directory + 'group2/', frac))
	length_group3 = max(__large_group_lengths(directory + 'group3/', frac))
	y = __create_y(length_group1, length_group2, length_group3)
	etc = ExtraTreesClassifier(n_estimators=100)
	etc.fit(data, y)
	print('Feature importance complete in %0.3fs' % (time() - t0))
	print(etc.feature_importances_)
	feat_importances = pd.Series(etc.feature_importances_, index=data.columns)
	feat_importances.nlargest(10).plot(kind='barh')
	plt.show()


if __name__ == "__main__":
	# Prints out the amounts of a specific feature that has been collected
	print(__group_lengths('../../data_output/group1/'))
	print(__group_lengths('../../data_output/group2/'))
	print(__group_lengths('../../data_output/group3/'))
	# Checks the importance of the features to decide what features are most important
	__check_importance('../../data_output/')
	# print(__large_group_lengths('../../data_output_further/group1/'))
	# print(__large_group_lengths('../../data_output_further/group2/'))
	# print(__large_group_lengths('../../data_output_further/group3/'))

	# Checks the importance of the features from a large dataset,
	# taking only a fraction of the data because it is too large
	# frac = 0.1
	# __large_check_importance('../../data_output_further/', frac)
