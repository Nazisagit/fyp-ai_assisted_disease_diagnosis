# Filename: ComponentAnalysis.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Created: 21/03/2019


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


def group1():
	width = pd.DataFrame(pd.read_csv('../data_output/group1/width.csv'))
	height = pd.DataFrame(pd.read_csv('../data_output/group1/height.csv'))
	area = pd.DataFrame(pd.read_csv('../data_output/group1/area.csv'))
	colour = pd.DataFrame(pd.read_csv('../data_output/group1/colour.csv'))
	length = pd.DataFrame(pd.read_csv('../data_output/group1/length.csv'))

	width_subset = width.iloc[1:370000]
	height_subset = height.iloc[1:370000]
	area_subset = area.iloc[1:370000]
	colour_subset = colour.iloc[1:370000]
	length_subset = length.iloc[1:370000]

	wh = width_subset.join(height_subset)
	wha = wh.join(area_subset)
	whac = wha.join(colour_subset)
	group_1 = whac.join(length_subset)

	return group_1


def group2():
	width = pd.DataFrame(pd.read_csv('../data_output/group2/width.csv'))
	height = pd.DataFrame(pd.read_csv('../data_output/group2/height.csv'))
	area = pd.DataFrame(pd.read_csv('../data_output/group2/area.csv'))
	colour = pd.DataFrame(pd.read_csv('../data_output/group2/colour.csv'))
	length = pd.DataFrame(pd.read_csv('../data_output/group2/length.csv'))

	width_subset = width.loc[1:1700000]
	height_subset = height.loc[1:1700000]
	area_subset = area.loc[1:1700000]
	colour_subset = colour.loc[1:1700000]
	length_subset = length.loc[1:1700000]

	wh = width_subset.join(height_subset)
	wha = wh.join(area_subset)
	whac = wha.join(colour_subset)
	group_2 = whac.join(length_subset)

	return group_2


def group3():
	width = pd.DataFrame(pd.read_csv('../data_output/group3/width.csv'))
	height = pd.DataFrame(pd.read_csv('../data_output/group3/height.csv'))
	area = pd.DataFrame(pd.read_csv('../data_output/group3/area.csv'))
	colour = pd.DataFrame(pd.read_csv('../data_output/group3/colour.csv'))
	length = pd.DataFrame(pd.read_csv('../data_output/group3/length.csv'))

	width_subset = width.loc[1:4600000]
	height_subset = height.loc[1:4600000]
	area_subset = area.loc[1:4600000]
	colour_subset = colour.loc[1:4600000]
	length_subset = length.loc[1:4600000]

	wh = width_subset.join(height_subset)
	wha = wh.join(area_subset)
	whac = wha.join(colour_subset)
	group_3 = whac.join(length_subset)

	return group_3


def pca_group_1(n_components=None):
	print('Group 1')
	pcag1 = PCA(n_components=n_components, svd_solver='randomized', iterated_power=10)
	pcag1.fit(group1().sample(n=100000))
	print('Variance ratio: ', pcag1.explained_variance_ratio_)
	print('Noice variance: ', pcag1.noise_variance_)
	print('Singular values: ', pcag1.singular_values_, '\n')


def pca_group_2(n_components=None):
	print('Group 2')
	pcag2 = PCA(n_components=n_components, svd_solver='randomized', iterated_power=10)
	pcag2.fit(group2().sample(n=100000))
	print('Variance ratio: ', pcag2.explained_variance_ratio_)
	print('Noise variance: ', pcag2.noise_variance_)
	print('Singular values: ', pcag2.singular_values_, '\n')


def pca_group_3(n_components=None):
	print('Group 3')
	pcag3 = PCA(n_components=n_components, svd_solver='randomized', iterated_power=10)
	pcag3.fit(group3().sample(n=100000))
	print('Variance ratio: ', pcag3.explained_variance_ratio_)
	print('Noise variance: ', pcag3.noise_variance_)
	print('Singular values: ', pcag3.singular_values_, '\n')


def pca_combined_data(n_components=None):
	print('Combined data')
	data = group1().append([group2(), group3()], ignore_index=True)
	pca = PCA(n_components=n_components, svd_solver='randomized', iterated_power=10)
	pca.fit(data)
	print('Variance ratio: ', pca.explained_variance_ratio_)
	print('Noice variance: ', pca.noise_variance_)
	print('Singular values: ', pca.singular_values_, '\n')


def analyse_all_components():
	print('PCA on all components')
	pca_group_1()
	pca_group_2()
	pca_group_3()
	pca_combined_data()


def analyse_four_components():
	print('PCA on 4 components')
	pca_group_1(4)
	pca_group_2(4)
	pca_group_3(4)
	pca_combined_data(4)


def check_importance():
	print('Checking feature importance')
	data = group1().append([group2(), group3()], ignore_index=True)
	y = create_y(len(group1()), len(group2()), len(group3()))
	etc = ExtraTreesClassifier()
	etc.fit(data, y)
	print(etc.feature_importances_)


def create_y(g1, g2, g3):
	y = [1 for i in range(g1)]
	y.extend([2 for i in range(g2)])
	y.extend([3 for i in range(g3)])
	return y


def main():
	analyse_all_components()
	analyse_four_components()
	check_importance()


if __name__ == "__main__":
	main()
