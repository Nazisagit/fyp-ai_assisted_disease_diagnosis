# Filename: ipcl_diagnosis.py
# Author: Dmytro Poliyivets
# Institution: King's College London
# Copyright: 2018, Dmytro Poliyivets, King's College London


from python.ipcl_diagnoser import IPCLDiagnoser
from python.feature_detection.image_extractor import ImageExtractor
from python.feature_detection.feature_detector import FeatureDetector
from pathlib import Path
from argparse import ArgumentParser


def diagnosis(images, patient_number, patient_date, amt, sample, max_iter):
	# directory of the original endoscopic images
	original_images = images + patient_number + '/' + patient_date + '/'
	# directory of the extracted images
	extracted_images = images + '../extracted_images/' + patient_number + '/' + patient_date + '/'
	# directory of the feature detected images
	detected_features = images + '../detected_features/' + patient_number + '/' + patient_date + '/'

	extract_images(original_images, extracted_images)
	feature_tables = detect_features(extracted_images, detected_features)

	ipcl_diagnoser = IPCLDiagnoser(feature_tables, detected_features)
	ipcl_diagnoser.create_feature_dataframe()

	# Provide sample_amt to tell the diagnoser how many samples to take the same amount
	# from each group
	# Provide the max_iter to tell the diagnoser the max number of iterations to perform
	# on the training set
	ipcl_diagnoser.diagnose(images, amt, sample, max_iter)


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
	return feature_detector.get_feature_tables()


def check_init_files(data_output, group):
	width = Path(data_output + group + 'width.csv')
	height = Path(data_output + group + 'height.csv')
	colour = Path(data_output + group + 'colour.csv')
	area = Path(data_output + group + 'area.csv')
	length = Path(data_output + group + 'length.csv')
	paths = [width, height, colour, area, length]
	for path in paths:
		return True if path.exists() else False


if __name__ == "__main__":
	diagnosis('../Student Data/', '0099053314d', '2017-06-12', [370000, 1700000, 4600000], 100000, 2000)
	parser = ArgumentParser(prog='python ipcl_diagnosis.py', description='IPCL diagnosis.')
	parser.add_argument('-i', '--images', help='Main directory containing all the patient folders.')
	parser.add_argument('-p', '--patient', help='Patient\'s folder number')
	parser.add_argument('-d', '--date', help='Patient\'s endoscopy date.')

