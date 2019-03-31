# Filename: trainer.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Created: 26/03/2019

import numpy as np
import pandas as pd
from time import time
from joblib import dump

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import LinearSVC

""" This module trains a classifier for classifying IPCL
	into either Group 1, 2, or 3.
"""


def train(input_dir, subset_size, sample_size, max_iter):
	x = __create_x(input_dir, subset_size, sample_size)
	y = __create_y(sample_size)

	x_train, x_test, y_train, y_test = __pca(x, y)

	clf = __linearsvc(x, y, max_iter)

	t2 = time()
	y_pred = clf.predict(x_test)
	print('Training classification prediction done in %0.3fs' % (time() - t2))
	arg_max_count, percentage, max_counts = __calculate_prediction_percentage(y_pred)
	print('Prediction: Group ', arg_max_count, ' at %0.4fs' % percentage, '%')
	print('Prediction: ', max_counts, '/', len(y_pred))
	print('Training classification prediction report: \n', classification_report(y_test, y_pred))

	# dump(clf, './clf-kk.joblib')


def __calculate_prediction_percentage(prediction):
	counts = np.bincount(prediction)
	arg_max_count = np.argmax(counts)
	percentage = (max(counts)/len(prediction)) * 100
	return arg_max_count, percentage, max(counts)


def __neigh(x, y):
	t0 = time()
	neigh = KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='auto', n_jobs=2)
	neigh.fit(x, y)
	print('Fitting done in %0.3fs' % (time() - t0), '\n')
	return neigh


def __linearsvc(x, y, max_iter):
	t0 = time()
	clf = LinearSVC(random_state=20, multi_class='ovr', max_iter=max_iter, penalty='l2')
	clf.fit(x, y)
	print('Fitting done in %0.3fs' % (time() - t0), '\n')
	return clf


def __pca(x, y):
	t0 = time()
	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=0.25, random_state=1)
	pca = PCA(svd_solver='full')
	pca.fit(x_train)
	x_train_pca = pca.transform(x_train)
	x_test_pca = pca.transform(x_test)
	print('\nPrincipal component analysis done in %0.3fs.' % (time() - t0))
	return x_train_pca, x_test_pca, y_train, y_test


def __kbins(x, y):
	t0 = time()
	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=0.25, random_state=1)
	est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
	x_train_est = est.fit_transform(x_train, y_train)
	x_test_est = est.fit_transform(x_test, y_test)
	print('\nK-bins discretization done in %0.3fs.' % (time() - t0))
	return x_train_est, x_test_est, y_train, y_test


def __create_x(directory, subset_size, sample_size):
	group1_subset = __group_subset(__group(directory + 'group1/'), subset_size[0])
	group2_subset = __group_subset(__group(directory + 'group2/'), subset_size[1])
	group3_subset = __group_subset(__group(directory + 'group3/'), subset_size[2])
	group1_subset_sample = group1_subset.sample(n=sample_size, random_state=10)
	group2_subset_sample = group2_subset.sample(n=sample_size, random_state=20)
	group3_subset_sample = group3_subset.sample(n=sample_size, random_state=40)
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
	max_iter = 1500
	train(input_dir, subset_size, sample_size, max_iter)
