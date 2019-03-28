# Filename: ipcl_diagnoser.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Created: 21/03/2019

import numpy as np
from time import time
from joblib import load


def diagnose(features_df, n_components):
	clf = load('./clf.joblib')
	t0 = time()
	prediction = clf.predict(__return_features(features_df, n_components))
	print('Classification prediction done in %0.3fs' % (time() - t0), '\n')
	print('Prediction: ', prediction)
	max_pred_class, percentage = __calculate_prediction_percentage(prediction)
	print('Prediction: Group ', max_pred_class, ' at %0.4fs' % percentage, '%')


def __calculate_prediction_percentage(prediction):
	counts = np.bincount(prediction)
	max_count = np.argmax(counts)
	percentage = (max(counts)/len(prediction)) * 100
	return max_count, percentage


def __return_features(features_df, n_components):
	if n_components is None:
		return features_df
	else:
		return features_df.iloc[:, 0:n_components]

