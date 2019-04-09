# Filename: ipcl_diagnosis.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Last modified: 07/04/2019

"""
Performs diagnosis, either with Dmitry's naive Bayes classifier,
linear support vector classifier with k-bins discretizer pre-processing,
or gradient boosting classifier.
"""


import pandas as pd

from source.diagnosis.ipcl_diagnoser import diagnose
from source.diagnosis.Diagnoser import Diagnoser
from source.common.common import extract_images, detect_features


def __create_feature_dataframe(feature_tables):
	"""
	Creates a pandas DataFrame of all the features
	:param feature_tables: list of feature tables; each feature table represents
							one image
	:return: returns a pandas DataFrame of all the features
	"""
	width_list = list()
	height_list = list()
	area_list = list()
	red_list = list()
	blue_list = list()
	green_list = list()
	length_list = list()

	for table in feature_tables:
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

	features_df = pd.DataFrame({
		'Width': width_list,
		'Height': height_list,
		'Area': area_list,
		'Red': red_list,
		'Green': green_list,
		'Blue': blue_list,
		'Length': length_list
	})
	return features_df


def __naive_bayes(feature_tables):
	"""
	:param feature_tables: features to be used in the naive Bayes
	diagnosis
	"""
	bayes_diagnoser = Diagnoser(feature_tables)
	bayes_diagnoser.analyse_feature_table()
	bayes_diagnoser.naive_bayes()


def diagnosis(images, patient_number, patient_date, classifier):
	"""
	Diagnoses a patient based on their endoscopic images
	:param images: folder where all the image folders should be
	:param patient_number: folder of a patient
	:param patient_date: folder of the endoscopic images on the date taken
	:param classifier: the full path to a classifier
	"""
	# folder of the original endoscopic images
	original_images = images + patient_number + '/' + patient_date + '/'
	# folder of the extracted images
	extracted_images = images + 'extracted_images/' + patient_number + '/' + patient_date + '/'
	# folder of the feature detected images
	detected_features = images + 'detected_features/' + patient_number + '/' + patient_date + '/'

	extract_images(original_images, extracted_images)
	feature_tables = detect_features(extracted_images, detected_features)
	if classifier is None:
		# Diagnosis using Dmitry's Naive Bayes Classifier
		__naive_bayes(feature_tables)
	else:
		feature_dataframe = __create_feature_dataframe(feature_tables)
		diagnose(feature_dataframe, classifier)


if __name__ == "__main__":
	# Provide the full path to folder of images
	images = 'D:/University/FYP/fyp-ai_assisted_disease_diagnosis/images/cleaned_test/'
	# Provide the patient ID
	patient_number = '0017021777d'
	# Provide the date of the images
	patient_date = '2017-03-07'
	# Provide the full path to the classifier
	classifier = 'D:/University/FYP/fyp-ai_assisted_disease_diagnosis/source/classifiers/gbc-c26.joblib'
	# If you provide classifier as None, Dmitry's naive Bayes classifier will be used
	diagnosis(images, patient_number, patient_date, classifier)
