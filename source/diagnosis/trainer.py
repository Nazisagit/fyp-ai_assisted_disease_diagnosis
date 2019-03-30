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
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import LinearSVC

""" This module trains a classifier for classifying IPCL
	into either Group 1, 2, or 3.
"""


def train(input_dir, sample_size, max_iter):
	print('Beginning training.')
	x = __create_x(input_dir, sample_size)
	y = __create_y(sample_size)

	x_train_pca, x_test_pca, y_train, y_test = __kbins(x, y)

	t1 = time()
	clf = LinearSVC(random_state=20, multi_class='ovr', max_iter=max_iter, penalty='l2')
	clf.fit(x_train_pca, y_train)
	print('Fitting done in %0.3fs.\n' % (time() - t1))

	t2 = time()
	y_pred = clf.predict(x_test_pca)
	print('Training classification prediction done in %0.3fs.' % (time() - t2))
	print('Training prediction: ', y_pred)
	print('Training classification prediction report: \n', classification_report(y_test, y_pred))

	dump(clf, './clf-further.joblib')


def __pca(x, y):
	t0 = time()
	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=0.25, random_state=1)
	pca = PCA(svd_solver='full')
	x_train_filled = x_train.fillna(x_train.mean())
	x_test_filled = x_test.fillna(x_test.mean())
	pca.fit(x_train_filled)
	x_train_pca = pca.transform(x_train_filled)
	x_test_pca = pca.transform(x_test_filled)
	print('\nPrincipal component analysis done in %0.3fs.' % (time() - t0))
	return x_train_pca, x_test_pca, y_train, y_test


def __kbins(x, y):
	t0 = time()
	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=0.25, random_state=1)
	est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
	x_train_filled = x_train.fillna(x_train.mean())
	x_test_filled = x_test.fillna(x_test.mean())
	x_train_est = est.fit_transform(x_train_filled, y_train)
	x_test_est = est.fit_transform(x_test_filled, y_test)
	print('\nK-bins discretization done in %0.3fs.' % (time() - t0))
	return x_train_est, x_test_est, y_train, y_test


def __create_x(directory, sample_size):
	group_1_sample = __group(directory + 'group1/').sample(n=sample_size, random_state=20)
	group_2_sample = __group(directory + 'group2/').sample(n=sample_size, random_state=60)
	group_3_sample = __group(directory + 'group3/').sample(n=sample_size, random_state=40)
	x = group_1_sample.append([group_2_sample, group_3_sample], ignore_index=True)
	return x


def __create_y(sample_size):
	y = [1 for i in range(sample_size)]
	y.extend([2 for i in range(sample_size)])
	y.extend([3 for i in range(sample_size)])
	return y


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

	width = pd.concat(width_list)
	height = pd.concat(height_list)
	area = pd.concat(area_list)

	wh = width.join(height)
	group = wh.join(area)

	return group


if __name__ == '__main__':
	input_dir = '../../data_output/'
	sample_size = 200000
	max_iter = 1500
	train(input_dir, sample_size, max_iter)
