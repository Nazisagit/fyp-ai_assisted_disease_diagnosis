# Filename: trainer.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Created: 26/03/2019

import pandas as pd
from time import time
from joblib import dump

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def train(directory, amt, sample, max_iter):
	x = __create_x(directory, amt, sample)
	y = __create_y(sample)

	t0 = time()
	x_train_pca, x_test_pca, y_train, y_test = __pca(x, y)
	print('\nPrincipal component analysis done in %0.3fs' % (time() - t0))

	t1 = time()
	clf = LinearSVC(random_state=1, multi_class='ovr', max_iter=max_iter, penalty='l2')
	clf.fit(x_train_pca, y_train)
	print('Fitting done in %0.3fs' % (time() - t1), '\n')

	t2 = time()
	y_pred = clf.predict(x_test_pca)
	print('Training classification prediction done in %0.3fs' % (time() - t2))
	print('Training prediction: ', y_pred)
	print('Training classification prediction report: \n', classification_report(y_test, y_pred))

	dump(clf, './clf.joblib')


def __pca(x, y):
	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=0.25, random_state=1)
	pca = PCA(n_components=4, svd_solver='full')
	pca.fit(x_train)
	x_train_pca = pca.transform(x_train)
	x_test_pca = pca.transform(x_test)
	return x_train_pca, x_test_pca, y_train, y_test


def __create_x(directory, amt, sample):
	group1_subset = __group_subset(__group(directory + 'group1/'), amt[0])
	group2_subset = __group_subset(__group(directory + 'group2/'), amt[1])
	group3_subset = __group_subset(__group(directory + 'group3/'), amt[2])
	group1_subset_sample = group1_subset.sample(n=sample, random_state=1)
	group2_subset_sample = group2_subset.sample(n=sample, random_state=1)
	group3_subset_sample = group3_subset.sample(n=sample, random_state=1)
	x = group1_subset_sample.append([group2_subset_sample, group3_subset_sample], ignore_index=True)
	return x


def __create_y(amt):
	y = [1 for i in range(amt)]
	y.extend([2 for i in range(amt)])
	y.extend([3 for i in range(amt)])
	return y


def cross_validate(self, x):
	kf = KFold(n_splits=10)
	kf.get_n_splits(x)


def __group(directory):
	width = pd.DataFrame(pd.read_csv(directory + 'width.csv'))
	height = pd.DataFrame(pd.read_csv(directory + 'height.csv'))
	area = pd.DataFrame(pd.read_csv(directory + 'area.csv'))
	colour = pd.DataFrame(pd.read_csv(directory + 'colour.csv'))
	length = pd.DataFrame(pd.read_csv(directory + 'length.csv'))

	return [width, height, area, colour, length]


def __group_length(group):
	print('  Width  |  Height  |  Area  | Colour  | Length')
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
