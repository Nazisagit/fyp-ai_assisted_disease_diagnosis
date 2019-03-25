# Filename: Main.py
# Author: Dmytro Poliyivets
# Institution: King's College London
# Copyright: 2018, Dmytro Poliyivets, King's College London


# Import necessary classes:
from python.TypeDiagnosis import TypeDiagnosis
from python.GroupDiagnosis import GroupDiagnosis
from python.Diagnosis import Diagnosis
from python.ImageExtractor import ImageExtractor
from python.FeatureDetector import FeatureDetector
from processing.DataCollecting import DataCollecting
import os
from pathlib import Path

# TODO
# Diagnosis is not efficient enough and takes too long


# Application main method
def main():
	type1 = 'type1/'
	type2 = 'type2/'
	type3 = 'type3/'
	type4 = 'type4/'
	type5 = 'type5/'
	group1 = 'group1/'
	group2 = 'group2/'
	group3 = 'group3/'

	patient_number = '0099053314d'
	patient_date = '2017-06-12'
	original_images = '../Student Data/' + patient_number + '/' + patient_date + '/'
	extracted_images = '../extracted_images/' + patient_number + '/' + patient_date + '/'
	detected_features = '../detected_features/' + patient_number + '/' + patient_date + '/'
	data_output = '../data_output/'

	# extract_images(original_images, extracted_images)
	# feature_tables, number_ipcls = detect_features(extracted_images, detected_features)

	# data_collecting = DataCollecting(feature_tables, number_ipcls, data_output, group3)
	# Comment out this command after collecting data for your first image
	# if check_init_files(data_output):
	#   data_collecting.init_output_files()
	# data_collecting.save_data()

	# diagnosis = Diagnosis(feature_tables, number_ipcls, detected_features)
	# diagnosis.create_feature_dataframe()
	# # train classifier with n samples and max iterations
	# classifier = diagnosis.train(200000, 2000)
	# diagnosis.diagnose(classifier)


def extract_images(original_images, extracted_images):
	path = Path(extracted_images)
	if not path.exists():
		path.mkdir()
	image_extractor = ImageExtractor(original_images, extracted_images)
	image_extractor.extract()


def detect_features(extracted_images, detected_features):
	path = Path(detected_features)
	if not path.exists():
		path.mkdir()
	feature_detector = FeatureDetector(extracted_images, detected_features)
	feature_detector.run()
	return feature_detector.get_feature_tables(), feature_detector.get_number_ipcls()


def check_init_files(data_output):
	width = Path(data_output + 'width.csv')
	height = Path(data_output + 'height.csv')
	colour = Path(data_output + 'colour.csv')
	area = Path(data_output + 'area.csv')
	length = Path(data_output + 'length.csv')
	paths = [width, height, colour, area, length]
	for path in paths:
		return True if path.exists() else False


# Call main function
if __name__ == "__main__":
	main()

