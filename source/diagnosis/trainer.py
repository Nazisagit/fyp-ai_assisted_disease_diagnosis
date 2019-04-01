# Filename: trainer.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Created: 26/03/2019

import numpy as np
import pandas as pd
from time import time
from joblib import dump

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import LinearSVC

""" This module trains a classifier for classifying IPCL
	into either Group 1, 2, or 3.
"""


def train(input_dir, sample_size):
	x = __create_x(input_dir, sample_size)
	y = __create_y(sample_size)

	x_train, x_test, y_train, y_test = __split(x, y)

	# x_train_pca, x_test_pca = __pca(x_train, x_test)
	x_train_est, x_test_est = __kbins(x_train, x_test, y_train, y_test)

	# clf = __linearsvc(x_train_pca, y_train_pca, max_iter)
	clf = __linearsvc(x_train_est, y_train)

	t0 = time()
	# y_pred = clf.predict(x_test_pca)
	y_pred = clf.predict(x_test_est)
	print('Training classification prediction done in %0.3fs' % (time() - t0))
	print('Training classification prediction report: \n', classification_report(y_test, y_pred))

	# dump(clf, './clf-kk.joblib')


def __linearsvc(x, y):
	t0 = time()
	clf = LinearSVC(random_state=0, multi_class='ovr', max_iter=2000, penalty='l2')
	clf.fit(x, y)
	print('Fitting LinearSVC done in %0.3fs' % (time() - t0), '\n')
	return clf


def __neigh(x, y):
	t0 = time()
	neigh = KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='auto', n_jobs=2)
	neigh.fit(x, y)
	print('Fitting K-Neightbours Classifier done in %0.3fs' % (time() - t0), '\n')
	return neigh


def __gbc(x, y):
	t0 = time()
	gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
	gbc.fit(x, y)
	print('Fitting Gradient Boosting Classifier done in %0.3fs' % (time() - t0), '\n')
	return gbc


def __pca(x_train, x_test):
	t0 = time()
	pca = PCA(svd_solver='full')
	pca.fit(x_train)
	x_train_pca = pca.transform(x_train)
	x_test_pca = pca.transform(x_test)
	print('\nPrincipal component analysis done in %0.3fs.' % (time() - t0))
	return x_train_pca, x_test_pca


def __kbins(x_train, x_test, y_train, y_test):
	t0 = time()
	est = KBinsDiscretizer(n_bins=5, encode='onehot', strategy='uniform')
	x_train_est = est.fit_transform(x_train, y_train)
	x_test_est = est.fit_transform(x_test, y_test)
	print('\nK-bins discretization pre-processing done in %0.3fs.' % (time() - t0))
	return x_train_est, x_test_est


def __split(x, y):
	x_train, x_test, y_train, y_test = train_test_split(
		x, y, random_state=0, test_size=0.25)
	return x_train, x_test, y_train, y_test


def __create_x(directory, sample_size):
	group1 = __group(directory + 'group1/')
	group2 = __group(directory + 'group2/')
	group3 = __group(directory + 'group3/')
	group1_sample = group1.sample(n=sample_size, random_state=0)
	group2_sample = group2.sample(n=sample_size, random_state=0)
	group3_sample = group3.sample(n=sample_size, random_state=0)
	x = group1_sample.append([group2_sample, group3_sample], ignore_index=True)
	return x


def __create_y(amt):
	y = [1 for i in range(amt)]
	y.extend([2 for i in range(amt)])
	y.extend([3 for i in range(amt)])
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

	colour_chunks = pd.read_csv(directory + 'colour.csv', chunksize=chunksize)
	colour_list = list()
	for chunk in colour_chunks:
		colour_list.append(chunk)

	# length_chunks = pd.read_csv(directory + 'length.csv', chunksize=chunksize)
	# length_list = list()
	# for chunk in length_chunks:
	# 	length_list.append(chunk)

	width = pd.concat(width_list)
	height = pd.concat(height_list)
	area = pd.concat(area_list)
	colour = pd.concat(colour_list)
	# length = pd.concat(length_list)

	# group = width.join(height)
	# group = group.join(area)
	# group = group.join(colour)
	# group = group.join(length)
	group = width.join(colour)

	return group


if __name__ == '__main__':
	input_dir = '../../data_output/'
	sample_size = 200000
	train(input_dir, sample_size)
