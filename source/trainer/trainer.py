# Filename: trainer.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Last modified: 06/04/2019

"""
Creates a trained classifier: either a
linear support vector classifier with k-bins discretizer pre-processing,
or a gradient boosting classifier.
"""

import pandas as pd
from time import time
from joblib import dump

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import LinearSVC


def train(input_dir, sample_size, dump_dir ,classifier_name):
	"""
	Trains either a linear support vector classifier with k-bins discretizer
	pre-processing or gradient boosting classifier.
	:param input_dir: folder where the group folders area
	:param sample_size: sample to be taken from each group
	:param features: list of features to include
	:param classifier_name: name of the classifier
	https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization_classification.html
	"""
	print('Beginning training\n')
	print('Creating samples\n')
	x = __create_x(input_dir, sample_size)
	y = __create_y(sample_size)
	print('Created samples\n')

	x_train, x_test, y_train, y_test = __split(x, y)

	if 'lsvc' in classifier_name:
		x_train_est, x_test_est = __kbins(x_train, x_test, y_train, y_test)
		lsvc = __linearsvc(x_train_est, y_train)

		t0 = time()
		y_pred = lsvc.predict(x_test_est)
		print('Training classification prediction done in %0.3fs.\n' % (time() - t0))
		print('Training classification prediction report: \n', classification_report(y_test, y_pred))
		dump(lsvc, '{}/{}.joblib'.format(dump_dir, classifier_name))
	elif 'gbc' in classifier_name:
		gbc = __gbc(x_train, y_train)

		t0 = time()
		y_pred = gbc.predict(x_test)
		print('Training classification prediction done in %0.3fs.\n' % (time() - t0))
		print('Training classification prediction report: \n', classification_report(y_test, y_pred))

		dump(gbc, '{}/{}.joblib'.format(dump_dir, classifier_name))


def __linearsvc(x, y):
	"""
	Fits the linear support vector classifier to the data
	:param x: features
	:param y: classes
	:return: fitted LinearSVC
	https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
	"""
	t0 = time()
	lsvc = LinearSVC(random_state=0, multi_class='ovr', max_iter=2500, penalty='l2')
	lsvc.fit(x, y)
	print('Fitting LinearSVC done in %0.3fs' % (time() - t0), '\n')
	return lsvc


def __kbins(x_train, x_test, y_train, y_test):
	"""
	Performs data pre-processing using k-bins discretizer
	:param x_train: features for training classifier
	:param x_test: features for testing classifier
	:param y_train: classes for training classifier
	:param y_test: classes for testing classifier
	:return: fitted x_train and x_test
	"""
	print('K-bins discretization pre-processing beginning.\n')
	t0 = time()
	est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
	x_train_est = est.fit_transform(x_train, y_train)
	x_test_est = est.fit_transform(x_test, y_test)
	print('K-bins discretization pre-processing done in %0.3fs.\n' % (time() - t0))
	return x_train_est, x_test_est


def __gbc(x, y):
	"""
	Fits the GradientBoostingClassifier to the data
	:param x: features
	:param y: classes
	:return: fitted GradientBoostingClassifier
	https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
	"""
	print('GradientBoostingClassifier fitting beginning.\n')
	t0 = time()

	gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.9, max_depth=20, random_state=0)
	gbc.fit(x, y)
	print('Fitting GradientBoostingClassifier done in %0.3fs.\n' % (time() - t0))
	return gbc


def __split(x, y):
	"""
	Splits the data into train and test sets
	:param x: features
	:param y: classes
	:return: x_train, x_test, y_train, y_test
	"""
	x_train, x_test, y_train, y_test = train_test_split(
		x, y, random_state=40, test_size=0.25)
	return x_train, x_test, y_train, y_test


def __create_x(directory, sample_size):
	"""
	Creates the features to be used for training
	:param directory: where the IPCL group folders should be
	:param sample_size: the sample size to be taken from each group
	:param features: list of features to include
	:return: a pandas DataFrame of 3 * sample_size
	"""
	group1 = __group(directory + 'group1/')
	group2 = __group(directory + 'group2/')
	group3 = __group(directory + 'group3/')
	group1_sample = group1.sample(n=sample_size, random_state=40)
	group2_sample = group2.sample(n=sample_size, random_state=40)
	group3_sample = group3.sample(n=sample_size, random_state=40)
	x = group1_sample.append([group2_sample, group3_sample], ignore_index=True)
	return x


def __create_y(amt):
	"""
	Creates the classes to be used for training
	:param amt: the number of classes needed for each group
	:return: a list of classes
	"""
	y = [1 for i in range(amt)]
	y.extend([2 for i in range(amt)])
	y.extend([3 for i in range(amt)])
	return y


def __group(group_folder):
	"""
	Groups together the data from the csv feature files in one group folder.
	Loads them in chunksizes for speed.
	:param group_folder: where the csv feature files should be
	:param features: list of features to include
	:return: a pandas DataFrame of the grouped data
	"""
	chunksize = 10 ** 6

	width_chunks = pd.read_csv(group_folder + 'width.csv', chunksize=chunksize)
	width_list = list()
	for chunk in width_chunks:
		width_list.append(chunk)

	height_chunks = pd.read_csv(group_folder + 'height.csv', chunksize=chunksize)
	height_list = list()
	for chunk in height_chunks:
		height_list.append(chunk)

	area_chunks = pd.read_csv(group_folder + 'area.csv', chunksize=chunksize)
	area_list = list()
	for chunk in area_chunks:
		area_list.append(chunk)

	colour_chunks = pd.read_csv(group_folder + 'colour.csv', chunksize=chunksize)
	colour_list = list()
	for chunk in colour_chunks:
		colour_list.append(chunk)

	length_chunks = pd.read_csv(group_folder + 'length.csv', chunksize=chunksize)
	length_list = list()
	for chunk in length_chunks:
		length_list.append(chunk)

	width = pd.concat(width_list)
	height = pd.concat(height_list)
	area = pd.concat(area_list)
	colour = pd.concat(colour_list)
	length = pd.concat(length_list)

	group = width.join(height)
	group = group.join(area)
	group = group.join(colour)
	# group = group.join(length)

	return group


if __name__ == '__main__':
	input_dir = 'D:/University/FYP/project/data_output/'
	sample_size = 100000
	dump_dir = 'D:/University/FYP/project/source/classifiers'
	classifier_name = 'gbc-test'
	train(input_dir, sample_size, dump_dir, classifier_name)
