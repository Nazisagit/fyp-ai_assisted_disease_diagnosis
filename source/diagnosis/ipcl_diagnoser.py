# Filename: ipcl_diagnoser.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Created: 21/03/2019

import numpy as np
from time import time
from joblib import load


def diagnose(features_df):
	clf = load('./clf.joblib')
	t0 = time()
	prediction = clf.predict(features_df)
	print('Classification prediction done in %0.3fs' % (time() - t0), '\n')
	arg_max_count, percentage, max_counts = __calculate_prediction_percentage(prediction)
	print('Prediction: Group ', arg_max_count, ' at %0.4fs' % percentage, '%')
	print('Prediction: ', max_counts, '/', len(prediction))


def __calculate_prediction_percentage(prediction):
	counts = np.bincount(prediction)
	arg_max_count = np.argmax(counts)
	percentage = (max(counts)/len(prediction)) * 100
	return arg_max_count, percentage, max(counts)


def __return_features(features_df, n_features):
	if n_features is None:
		return features_df
	else:
		return features_df.iloc[:, 0:n_features]


