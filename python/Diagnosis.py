# Filename: Diagnosis.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Created: 21/03/2019

import pandas as pd
import numpy as np
from time import time

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold


class Diagnosis:

	def __init__(self, feature_tables, number_ipcls, statistical_diagnoses_output):
		self.feature_tables = feature_tables
		self.number_ipcls = number_ipcls
		self.statistical_diagnoses_output = statistical_diagnoses_output
		self.diagnoses = dict()
		self.statistics = dict()
		self.features_df = None

	def create_feature_dataframe(self):
		width_list = list()
		height_list = list()
		area_list = list()
		red_list = list()
		blue_list = list()
		green_list = list()
		length_list = list()

		for table in self.feature_tables:
			table_height = len(table)
			for feature in [0, 1, 2, 3, 4]:
				for row in range(0, table_height):
					if feature == 0:
						width_list.append(table[row][feature])
					elif feature == 1:
						height_list.append(table[row][feature])
					elif feature == 2:
						area_list.append(table[row][feature])
					elif feature == 3:
						red_list.append(table[row][feature][0])
						blue_list.append(table[row][feature][1])
						green_list.append(table[row][feature][2])
					elif feature == 4:
						length_list.append(table[row][feature])

		self.features_df = pd.DataFrame({
			'Width': width_list,
			'Height': height_list,
			'Red': red_list,
			'Green': green_list,
			# 'Blue': blue_list,
			# 'Length': length_list
		})

	def diagnose(self, clf):
		t0 = time()
		prediction = clf.predict(self.features_df)
		print('Classification prediction done in %0.3fs' % (time() - t0))
		max_pred_class, percentage = self.calculate_prediction_percentage(prediction)
		print('Prediction: ', max_pred_class, ' at ', percentage, '%')

	@staticmethod
	def calculate_prediction_percentage(prediction):
		counts = np.bincount(prediction)
		max_count = np.argmax(counts)
		percentage = (max(counts)/len(prediction)) * 100
		return max_count, percentage

	def train(self, sample_amt, max_iter):
		group1 = self.group1().sample(n=sample_amt, random_state=1)
		group2 = self.group2().sample(n=sample_amt, random_state=1)
		group3 = self.group3().sample(n=sample_amt, random_state=1)
		x = group1.append([group2, group3], ignore_index=True)
		y = self.create_y(sample_amt)

		t0 = time()
		x_train_pca, x_test_pca, y_train, y_test = self.pca(x, y)
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

		return clf

	@staticmethod
	def pca(x, y):
		x_train, x_test, y_train, y_test = train_test_split(
			x, y, test_size=0.25, random_state=1)
		pca = PCA(n_components=4, svd_solver='full')
		pca.fit(x_train)
		x_train_pca = pca.transform(x_train)
		x_test_pca = pca.transform(x_test)

		return x_train_pca, x_test_pca, y_train, y_test

	@staticmethod
	def create_y(amt):
		y = [1 for i in range(amt)]
		y.extend([2 for i in range(amt)])
		y.extend([3 for i in range(amt)])
		return y

	def cross_validate(self, x):
		kf = KFold(n_splits=10)
		kf.get_n_splits(x)


	@staticmethod
	def group1():
		width = pd.DataFrame(pd.read_csv('../data_output/group1/width.csv'))
		height = pd.DataFrame(pd.read_csv('../data_output/group1/height.csv'))
		area = pd.DataFrame(pd.read_csv('../data_output/group1/area.csv'))
		colour = pd.DataFrame(pd.read_csv('../data_output/group1/colour.csv'))
		length = pd.DataFrame(pd.read_csv('../data_output/group1/length.csv'))

		width_subset = width.iloc[1:370000]
		height_subset = height.iloc[1:370000]
		area_subset = area.iloc[1:370000]
		colour_subset = colour.iloc[1:370000]
		length_subset = length.iloc[1:370000]

		wh = width_subset.join(height_subset)
		wha = wh.join(area_subset)
		whac = wha.join(colour_subset)
		group_1 = whac.join(length_subset)

		return group_1

	@staticmethod
	def group2():
		width = pd.DataFrame(pd.read_csv('../data_output/group2/width.csv'))
		height = pd.DataFrame(pd.read_csv('../data_output/group2/height.csv'))
		area = pd.DataFrame(pd.read_csv('../data_output/group2/area.csv'))
		colour = pd.DataFrame(pd.read_csv('../data_output/group2/colour.csv'))
		length = pd.DataFrame(pd.read_csv('../data_output/group2/length.csv'))

		width_subset = width.loc[1:1700000]
		height_subset = height.loc[1:1700000]
		area_subset = area.loc[1:1700000]
		colour_subset = colour.loc[1:1700000]
		length_subset = length.loc[1:1700000]

		wh = width_subset.join(height_subset)
		wha = wh.join(area_subset)
		whac = wha.join(colour_subset)
		group_2 = whac.join(length_subset)

		return group_2

	@staticmethod
	def group3():
		width = pd.DataFrame(pd.read_csv('../data_output/group3/width.csv'))
		height = pd.DataFrame(pd.read_csv('../data_output/group3/height.csv'))
		area = pd.DataFrame(pd.read_csv('../data_output/group3/area.csv'))
		colour = pd.DataFrame(pd.read_csv('../data_output/group3/colour.csv'))
		length = pd.DataFrame(pd.read_csv('../data_output/group3/length.csv'))

		width_subset = width.loc[1:4600000]
		height_subset = height.loc[1:4600000]
		area_subset = area.loc[1:4600000]
		colour_subset = colour.loc[1:4600000]
		length_subset = length.loc[1:4600000]

		wh = width_subset.join(height_subset)
		wha = wh.join(area_subset)
		whac = wha.join(colour_subset)
		group_3 = whac.join(length_subset)

		return group_3

