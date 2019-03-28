# Filename: component_analyzer.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Created: 21/03/2019


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


def __group(directory):
	width = pd.DataFrame(pd.read_csv(directory + 'width.csv'))
	height = pd.DataFrame(pd.read_csv(directory + 'height.csv'))
	area = pd.DataFrame(pd.read_csv(directory + 'area.csv'))
	colour = pd.DataFrame(pd.read_csv(directory + 'colour.csv'))
	length = pd.DataFrame(pd.read_csv(directory + 'length.csv'))

	return [width, height, area, colour, length]


def __group_length(group):
	print('  Width  |  Heigth  |  Area  | Colour  | Length')
	print([len(group[0]), len(group[1]), len(group[2]), len(group[3]), len(group[4])])


def __group_subset(group, amt):
	width_subset = group[0].iloc[1:amt]
	height_subset = group[1].iloc[1:amt]
	area_subset = group[2].iloc[1:amt]
	colour_subset = group[3].iloc[1:amt]
	length_subset = group[4].iloc[1:amt]

	wh = width_subset.join(height_subset)
	wha = wh.join(area_subset)
	whac = wha.join(colour_subset)
	subset = whac.join(length_subset)

	return subset


def __create_y(len_g1, len_g2, len_g3):
	y = [1 for i in range(len_g1)]
	y.extend([2 for i in range(len_g2)])
	y.extend([3 for i in range(len_g3)])
	return y


def __pca_group_1(directory, amt, sample, n_components=None):
	print('PCA for Group 1')
	group_directory = directory + 'group1/'
	group1_subset = __group_subset(__group(group_directory), amt)
	pcag1 = PCA(n_components=n_components, svd_solver='full')
	pcag1.fit(group1_subset.sample(n=sample))
	print('Variance ratio: ', pcag1.explained_variance_ratio_)
	print('Noice variance: ', pcag1.noise_variance_)
	print('Singular values: ', pcag1.singular_values_, '\n')


def __pca_group_2(directory, amt, sample, n_components=None):
	print('PCA for Group 2')
	group_directory = directory + 'group2/'
	group2_subset = __group_subset(__group(group_directory), amt)
	pcag2 = PCA(n_components=n_components, svd_solver='full')
	pcag2.fit(group2_subset.sample(n=sample))
	print('Variance ratio: ', pcag2.explained_variance_ratio_)
	print('Noise variance: ', pcag2.noise_variance_)
	print('Singular values: ', pcag2.singular_values_, '\n')


def __pca_group_3(directory, amt, sample, n_components=None):
	print('PCA for Group 3')
	group_directory = directory + 'group3/'
	group3_subset = __group_subset(__group(group_directory), amt)
	pcag3 = PCA(n_components=n_components, svd_solver='full')
	pcag3.fit(group3_subset.sample(n=sample))
	print('Variance ratio: ', pcag3.explained_variance_ratio_)
	print('Noise variance: ', pcag3.noise_variance_)
	print('Singular values: ', pcag3.singular_values_, '\n')


def __pca_combined_data(directory, amt, sample, n_components=None):
	print('PCA for combined data')
	group1_subset = __group_subset(__group(directory + 'group1/'), amt[0])
	group2_subset = __group_subset(__group(directory + 'group2/'), amt[1])
	group3_subset = __group_subset(__group(directory + 'group3/'), amt[2])
	data = group1_subset.append([group2_subset, group3_subset], ignore_index=True)
	pca = PCA(n_components=n_components, svd_solver='full')
	pca.fit(data.sample(n=sample))
	print('Variance ratio: ', pca.explained_variance_ratio_)
	print('Noice variance: ', pca.noise_variance_)
	print('Singular values: ', pca.singular_values_, '\n')


def __check_importance(directory, amt):
	print('Checking feature importance')
	group1_subset = __group_subset(__group(directory + 'group1/'), amt[0])
	group2_subset = __group_subset(__group(directory + 'group2/'), amt[1])
	group3_subset = __group_subset(__group(directory + 'group3/'), amt[2])
	data = group1_subset.append([group2_subset, group3_subset], ignore_index=True)
	y = __create_y(len(group1_subset), len(group2_subset), len(group3_subset))
	etc = ExtraTreesClassifier()
	etc.fit(data, y)
	print(etc.feature_importances_)


if __name__ == "__main__":
	# Prints out the amounts of a specific feature that has been collected
	__group_length(__group('../../data_output/group1/'))
	__group_length(__group('../../data_output/group2/'))
	__group_length(__group('../../data_output/group3/'))
	__pca_group_1('../../data_output/', amt=370000, sample=100000, n_components=None)
	__pca_group_2('../../data_output/', amt=1700000, sample=100000, n_components=None)
	__pca_group_3('../../data_output/', amt=4600000, sample=100000, n_components=None)
	__pca_combined_data('../../data_output/', [370000, 1700000, 4600000], 1500000, n_components=None)
	__check_importance('../../data_output/', [370000, 1700000, 4600000])
