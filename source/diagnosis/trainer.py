# Filename: trainer.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Created: 26/03/2019

import pandas as pd
from time import time
from joblib import dump

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

""" This module trains a classifier for classifying IPCL
	into either Group 1, 2, or 3.
"""


def train(input_dir, subset_size, sample_size, max_iter, n_components):
	x = __create_x(input_dir, subset_size, sample_size)
	y = __create_y(sample_size)

	t0 = time()
	x_train_pca, x_test_pca, y_train, y_test = __pca(x, y, n_components)
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


def __pca(x, y, n_components):
	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=0.25, random_state=1)
	pca = PCA(n_components=n_components, svd_solver='full')
	pca.fit(x_train)
	x_train_pca = pca.transform(x_train)
	x_test_pca = pca.transform(x_test)
	return x_train_pca, x_test_pca, y_train, y_test


def __create_x(directory, subset_size, sample_size):
	group1_subset = __group_subset(__group(directory + 'group1/'), subset_size[0])
	group2_subset = __group_subset(__group(directory + 'group2/'), subset_size[1])
	group3_subset = __group_subset(__group(directory + 'group3/'), subset_size[2])
	group1_subset_sample = group1_subset.sample(n=sample_size, random_state=1)
	group2_subset_sample = group2_subset.sample(n=sample_size, random_state=1)
	group3_subset_sample = group3_subset.sample(n=sample_size, random_state=1)
	x = group1_subset_sample.append([group2_subset_sample, group3_subset_sample], ignore_index=True)
	return x


def __create_y(amt):
	y = [1 for i in range(amt)]
	y.extend([2 for i in range(amt)])
	y.extend([3 for i in range(amt)])
	return y


def __group(directory):
	width = pd.DataFrame(pd.read_csv(directory + 'width.csv'))
	colour = pd.DataFrame(pd.read_csv(directory + 'colour.csv'))
	length = pd.DataFrame(pd.read_csv(directory + 'length.csv'))
	group = [width, colour, length]

	return group


def __group_subset(group, amt):
	width_subset = group[0].iloc[1:amt]
	colour_subset = group[1].iloc[1:amt]
	length_subset = group[2].iloc[1:amt]

	wc = width_subset.join(colour_subset)
	subset = wc.join(length_subset)

	return subset


if __name__ == '__main__':
	input_dir = '../../data_output/'
	subset_size = [370000, 1700000, 4600000]
	sample_size = 150000
	max_iter = 3000
	n_components = None
	train(input_dir, subset_size, sample_size, max_iter, n_components)