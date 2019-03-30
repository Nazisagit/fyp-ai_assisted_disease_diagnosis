# Filename: ipcl_diagnosis.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Last modified: 30/03/2019


import os
import pandas as pd

import source.diagnosis.ipcl_diagnoser as ipcl_diagnoser
from source.feature_detection.image_extractor import ImageExtractor
from source.feature_detection.FeatureDetector import FeatureDetector
from source.feature_detection.FurtherExtractor import FurtherExtractor


def create_feature_dataframe(feature_tables):
	width_list = list()
	height_list = list()
	area_list = list()
	red_list = list()
	blue_list = list()
	green_list = list()

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

	features_df = pd.DataFrame({
		'Width': width_list,
		'Height': height_list,
		'Area': area_list,
		'Red': red_list,
		'Green': green_list,
		'Blue': blue_list,
	})
	return features_df


def diagnosis(input_dir, patient_number, patient_date):
	# directory of the original endoscopic images
	original_images = input_dir + patient_number + '/' + patient_date + '/'
	# directory of the extracted images
	extracted_images = input_dir + 'extracted_images/' + patient_number + '/' + patient_date + '/'
	# directory of further extracted images
	further_extracted_images = input_dir + 'further_extracted/' + patient_number + '/' + patient_date + '/'
	# directory of the feature detected images
	detected_features = input_dir + 'detected_features/' + patient_number + '/' + patient_date + '/'

	__extract_images(original_images, extracted_images)
	__further_extract(extracted_images, further_extracted_images)
	feature_tables = __detect_features(further_extracted_images, detected_features)
	feature_dataframe = create_feature_dataframe(feature_tables)

	ipcl_diagnoser.diagnose(feature_dataframe)


def __extract_images(original_images, extracted_images):
	if not os.path.exists(extracted_images):
		os.makedirs(extracted_images)
	image_extractor = ImageExtractor(original_images, extracted_images)
	image_extractor.extract()


def __further_extract(extracted_images, further_extracted_images):
	if not os.path.exists(further_extracted_images):
		os.makedirs(further_extracted_images)
	further_extractor = FurtherExtractor(extracted_images, further_extracted_images)
	further_extractor.run()


def __detect_features(extracted_images, detected_features):
	if not os.path.exists(detected_features):
		os.makedirs(detected_features)
	feature_detector = FeatureDetector(extracted_images, detected_features)
	feature_detector.run()
	return feature_detector.get_feature_tables()


if __name__ == "__main__":
	input_dir = '../../Student Data/'
	patient_number = '0014059011d'
	patient_date = '2017-07-06'
	diagnosis(input_dir, patient_number, patient_date)
