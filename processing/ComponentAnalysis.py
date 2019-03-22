import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def group1():
	area = pd.DataFrame(pd.read_csv('../data_output/group1/area.csv'))
	colour = pd.DataFrame(pd.read_csv('../data_output/group1/colour.csv'))
	height = pd.DataFrame(pd.read_csv('../data_output/group1/height.csv'))
	length = pd.DataFrame(pd.read_csv('../data_output/group1/length.csv'))
	occurrences = pd.DataFrame(pd.read_csv('../data_output/group1/occurrences.csv'))
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


def group2():
	area = pd.DataFrame(pd.read_csv('../data_output/group2/area.csv'))
	colour = pd.DataFrame(pd.read_csv('../data_output/group2/colour.csv'))
	height = pd.DataFrame(pd.read_csv('../data_output/group2/height.csv'))
	length = pd.DataFrame(pd.read_csv('../data_output/group2/length.csv'))
	occurrences = pd.DataFrame(pd.read_csv('../data_output/group2/occurrences.csv'))
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


def group3():
	area = pd.DataFrame(pd.read_csv('../data_output/group3/area.csv'))
	colour = pd.DataFrame(pd.read_csv('../data_output/group3/colour.csv'))
	height = pd.DataFrame(pd.read_csv('../data_output/group3/height.csv'))
	length = pd.DataFrame(pd.read_csv('../data_output/group3/length.csv'))
	occurrences = pd.DataFrame(pd.read_csv('../data_output/group3/occurrences.csv'))
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


def analyse_all_components():
	print('PCA on all components')
	pcag1 = PCA(svd_solver='full')
	pcag1.fit(group1().sample(n=100000))
	print('Variance ratio: ', pcag1.explained_variance_ratio_)
	print('Noice variance: ', pcag1.noise_variance_, '\n')

	pcag2 = PCA(svd_solver='full')
	pcag2.fit(group2().sample(n=100000))
	print('Variance ratio: ', pcag2.explained_variance_ratio_)
	print('Noise variance: ', pcag2.noise_variance_, '\n')

	pcag3 = PCA(svd_solver='full')
	pcag3.fit(group3().sample(n=100000))
	print('Variance ratio: ', pcag3.explained_variance_ratio_)
	print('Noise variance: ', pcag3.noise_variance_, '\n')


def analyse_four_components():
	print('PCA on first 4 components: Width, Height, Red, Green')
	pcag1 = PCA(n_components=4, svd_solver='full')
	pcag1.fit(group1().sample(n=100000))
	print('Variance ratio: ', pcag1.explained_variance_ratio_)
	print('Noice variance: ', pcag1.noise_variance_, '\n')

	pcag2 = PCA(n_components=4, svd_solver='full')
	pcag2.fit(group2().sample(n=100000))
	print('Variance ratio: ', pcag2.explained_variance_ratio_)
	print('Noise variance: ', pcag2.noise_variance_, '\n')

	pcag3 = PCA(n_components=4, svd_solver='full')
	pcag3.fit(group3().sample(n=100000))
	print('Variance ratio: ', pcag3.explained_variance_ratio_)
	print('Noise variance: ', pcag3.noise_variance_, '\n')


def main():
	analyse_all_components()
	analyse_four_components()


if __name__ == "__main__":
	main()
