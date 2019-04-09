# Filename: ipcl_diagnoser.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Last modified: 08/04/2019

"""
Laods a classifier: either a linear support vector classifier
with k-bins discretizer, or a gradient boosting classifier
to diagnose and predict the IPCL group and level of cancer.
"""

import numpy as np
from collections import Counter
from time import time
from joblib import load


def __calculate_most_likely(prediction):
	"""
	Calculates and prints the percentage of the class with the highest
	number of predictions
	:param prediction: array of predictions
	"""
	counts = np.bincount(prediction)
	arg_max_count = np.argmax(counts)
	percentage = (max(counts)/len(prediction)) * 100
	print('Most likely: Group ', arg_max_count, ' at %0.3f' % percentage, '%.', max(counts), '/', len(prediction), '\n')
	__give_stage(arg_max_count)


def __calculate_second_likely(prediction):
	"""
	Calculates and prints the percentage of the class with the second
	highest number of predictions
	:param prediction: array of predictions
	https://stackoverflow.com/questions/31571520/how-can-i-find-the-second-most-common-number-in-an-array
	"""
	try:
		a = np.array(prediction)
		ctr = Counter(a.ravel())
		second_most_common_value, frequency = ctr.most_common(3)[1]
		percentage = (frequency/len(prediction)) * 100
		print('Second most likely: Group ', second_most_common_value, ' at %0.3f' % percentage, '%.', frequency, '/', len(prediction), '\n')
	except IndexError:
		print('No second group detected.\n')


def __calculate_least_likely(prediction):
	"""
	Calculates and prints the percentage of the class with the least
	number of predictions
	:param prediction: array of predictions
	https://stackoverflow.com/questions/31571520/how-can-i-find-the-second-most-common-number-in-an-array
	"""
	try:
		a = np.array(prediction)
		ctr = Counter(a.ravel())
		least_common_value, frequency = ctr.most_common(3)[2]
		percentage = (frequency / len(prediction)) * 100
		print('Least likely: Group ', least_common_value, ' at %0.3f' % percentage, '%.', frequency, '/', len(prediction), '\n')
	except IndexError:
		print('No third group detected.\n')


def __give_stage(value):
	if value == 1:
		print('Non-neoplastic\n')
	elif value == 2:
		print('Borderline\n')
	elif value == 3:
		print('Cancer\n')


def diagnose(features_df, classifier):
	"""
	Loads a classifier to diagnose the features DataFrame
	:param features_df: pandas DataFrame of features used
			to predict the IPCL group
	"""
	print('\n\nDISCLAIMER!: This software is experimental. \n'
	      'Please consult a qualified physician to obtain a diagnosis. \n')
	clf = load(classifier)
	t0 = time()
	prediction = clf.predict(features_df)
	__calculate_most_likely(prediction)
	__calculate_second_likely(prediction)
	__calculate_least_likely(prediction)
	print('Prediction completed in %0.3fs.\n' % (time() - t0))


