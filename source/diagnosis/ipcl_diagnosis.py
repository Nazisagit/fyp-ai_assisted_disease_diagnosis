# Filename: ipcl_diagnosis.py
# Author: Dmytro Poliyivets
# Institution: King's College London
# Copyright: 2018, Dmytro Poliyivets, King's College London


import os
import pandas as pd

import source.diagnosis.ipcl_diagnoser as ipcl_diagnoser
from source.feature_detection.image_extractor import ImageExtractor
from source.feature_detection.FeatureDetector import FeatureDetector


def create_feature_dataframe(feature_tables):
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
		'Red': red_list,
		'Green': green_list,
		'Blue': blue_list,
		'Length': length_list
	})
	return features_df


def diagnosis(input_dir, patient_number, patient_date):
	# directory of the original endoscopic images
	original_images = input_dir + patient_number + '/' + patient_date + '/'
	# directory of the extracted images
	extracted_images = input_dir + 'extracted_images/' + patient_number + '/' + patient_date + '/'
	# directory of the feature detected images
	detected_features = input_dir + 'detected_features/' + patient_number + '/' + patient_date + '/'

	extract_images(original_images, extracted_images)
	feature_tables = detect_features(extracted_images, detected_features)
	feature_dataframe = create_feature_dataframe(feature_tables)

	ipcl_diagnoser.diagnose(feature_dataframe)


def extract_images(original_images, extracted_images):
	if not os.path.exists(extracted_images):
		os.makedirs(extracted_images)
	image_extractor = ImageExtractor(original_images, extracted_images)
	image_extractor.extract()


def detect_features(extracted_images, detected_features):
	if not os.path.exists(detected_features):
		os.makedirs(detected_features)
	feature_detector = FeatureDetector(extracted_images, detected_features)
	feature_detector.run()
	return feature_detector.get_feature_tables()


if __name__ == "__main__":
	input_dir = '../../Student Data/'
	patient_number = '0014059011d'
	patient_date = '2017-07-06'
	n_features = None
	diagnosis(input_dir, patient_number, patient_date)
