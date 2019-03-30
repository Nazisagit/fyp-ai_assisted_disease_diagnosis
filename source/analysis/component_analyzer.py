# Filename: component_analyzer.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Created: 21/03/2019


import pandas as pd
import dask.dataframe as dd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


def __group(directory):
	width = pd.DataFrame(pd.read_csv(directory + 'width.csv'))
	height = pd.DataFrame(pd.read_csv(directory + 'height.csv'))
	area = pd.DataFrame(pd.read_csv(directory + 'area.csv'))

	wh = width.join(height)
	group = wh.join(area)

	return group


def __group_length(directory):
	width = __get_length(directory, 'width.csv')
	height = __get_length(directory, 'height.csv')
	area = __get_length(directory, 'area.csv')
	group = [width, height, area]

	return group


def __get_length(directory, feature):
	return len(dd.read_csv(directory + feature))


def __create_y(length):
	y = [1 for i in range(length)]
	y.extend([2 for i in range(length)])
	y.extend([3 for i in range(length)])
	return y


def __pca_group_1(directory, sample):
	print('PCA for Group 1')
	group_directory = directory + 'group1/'
	pcag1 = PCA(svd_solver='full')
	pcag1.fit(__group(group_directory).sample(n=sample, random_state=20))
	print('Variance ratio: ', pcag1.explained_variance_ratio_)
	print('Noice variance: ', pcag1.noise_variance_)
	print('Singular values: ', pcag1.singular_values_, '\n')


def __pca_group_2(directory, sample):
	print('PCA for Group 2')
	group_directory = directory + 'group2/'
	pcag2 = PCA(svd_solver='full')
	pcag2.fit(__group(group_directory).sample(n=sample, random_state=60))
	print('Variance ratio: ', pcag2.explained_variance_ratio_)
	print('Noise variance: ', pcag2.noise_variance_)
	print('Singular values: ', pcag2.singular_values_, '\n')


def __pca_group_3(directory, sample):
	print('PCA for Group 3')
	group_directory = directory + 'group3/'
	pcag3 = PCA(svd_solver='full')
	pcag3.fit(__group(group_directory).sample(n=sample, random_state=40))
	print('Variance ratio: ', pcag3.explained_variance_ratio_)
	print('Noise variance: ', pcag3.noise_variance_)
	print('Singular values: ', pcag3.singular_values_, '\n')


def __pca_combined_data(directory, sample):
	print('PCA for combined data')
	group_1_sample = __group(directory + 'group1/').sample(n=sample, random_state=20)
	group_2_sample = __group(directory + 'group2/').sample(n=sample, random_state=60)
	group_3_sample = __group(directory + 'group3/').sample(n=sample, random_state=40)
	data = group_1_sample.append([group_2_sample, group_3_sample], ignore_index=True)
	pca = PCA(svd_solver='full')
	pca.fit(data)
	print('Variance ratio: ', pca.explained_variance_ratio_)
	print('Noice variance: ', pca.noise_variance_)
	print('Singular values: ', pca.singular_values_, '\n')


def __check_importance(directory, sample):
	print('Checking feature importance')
	group_1_sample = __group(directory + 'group1/').sample(n=sample, random_state=20)
	group_2_sample = __group(directory + 'group2/').sample(n=sample, random_state=60)
	group_3_sample = __group(directory + 'group3/').sample(n=sample, random_state=40)
	data = group_1_sample.append([group_2_sample, group_3_sample], ignore_index=True)
	y = __create_y(sample)
	etc = ExtraTreesClassifier(n_estimators=100)
	etc.fit(data, y)
	print(etc.feature_importances_)


if __name__ == "__main__":
	# Prints out the amounts of a specific feature that has been collected
	# print(__group_length('../../data_output-further/group1/'))
	# print(__group_length('../../data_output-further/group2/'))
	# print(__group_length('../../data_output-further/group3/'))
	# __pca_group_1('../../data_output-further/', sample=250000)
	# __pca_group_2('../../data_output-further/', sample=250000)
	# __pca_group_3('../../data_output-further/', sample=250000)
	# __pca_combined_data('../../data_output-further/', 250000)
	__check_importance('../../data_output-further/', 250000)
