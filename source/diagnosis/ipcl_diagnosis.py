# Filename: ipcl_diagnosis.py
# Author: Dmytro Poliyivets
# Institution: King's College London
# Copyright: 2018, Dmytro Poliyivets, King's College London


import os
import pandas as pd

import source.diagnosis.ipcl_diagnoser as ipcl_diagnoser
from source.diagnosis.Diagnoser import Diagnoser
from source.feature_detection.image_extractor import extract
from source.feature_detection.FeatureDetector import FeatureDetector


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
		# 'Length': length_list
	})
	return features_df


def __extract_images(original_images, extracted_images):
	"""
	Uses the image_extractor module which is a modified version of
	Dmitry Poliyivets's FrameExtractor.
	Used to extract the regions of interest from the endoscopic images
	:param original_images: folder where the original images should be
	:param extracted_images: folder where the extracted images should be
								outputted to
	"""
	if not os.path.exists(extracted_images):
		os.makedirs(extracted_images)
	extract(original_images, extracted_images)


def __detect_features(extracted_images, detected_features):
	"""
	Slightly modified version of Dmitry Poliyivets's FeatureDetector
	used to detect the feature from the extracted regions of interest images
	:param extracted_images: folder where the extracted regions of interest images
								should be
	:param detected_features: folder where the images with detected features should be
	"""
	if not os.path.exists(detected_features):
		os.makedirs(detected_features)
	feature_detector = FeatureDetector(extracted_images, detected_features)
	feature_detector.run()
	return feature_detector.get_feature_tables()


def __naive_bayes(feature_tables):
	"""

	:param feature_tables:
	:return:
	"""
	bayes_diagnoser = Diagnoser(feature_tables)
	bayes_diagnoser.analyse_feature_table()
	bayes_diagnoser.naiveBayes()


def diagnosis(images, patient_number, patient_date):
	"""
	Diagnoses a patient based on their endoscopic images
	:param images: folder where all the image folders should be
	:param patient_number: folder of a patient
	:param patient_date: folder of the endoscopic images on the date taken
	"""
	# folder of the original endoscopic images
	original_images = images + patient_number + '/' + patient_date + '/'
	# folder of the extracted images
	extracted_images = images + 'extracted_images/' + patient_number + '/' + patient_date + '/'
	# folder of the feature detected images
	detected_features = images + 'detected_features/' + patient_number + '/' + patient_date + '/'

	__extract_images(original_images, extracted_images)
	feature_tables = __detect_features(extracted_images, detected_features)
	feature_dataframe = __create_feature_dataframe(feature_tables)

	# Diagnosis using Dmitry's Naive Bayes Classifier
	# __naive_bayes(feature_tables)

	ipcl_diagnoser.diagnose(feature_dataframe)


if __name__ == "__main__":
	input_dir = '../../images/'
	patient_number = '0017117424d'
	patient_date = '2017-10-17'
	diagnosis(input_dir, patient_number, patient_date)
