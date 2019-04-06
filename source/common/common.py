# Filename: common.py
# Author: Nazrin Pengiran
# Institution: King's College London
# Last modified: 06/04/2019

"""
Holds common functions to be used in other modules
"""

import os
from source.feature_detection.image_extractor import extract
from source.feature_detection.FeatureDetector import FeatureDetector


def extract_images(original_images, extracted_images):
	"""
	Used to extract the regions of interest from the endoscopic images
	:param original_images: folder where the original images should be
	:param extracted_images: folder where the extracted images should be
								outputted to
	"""
	if not os.path.exists(extracted_images):
		os.makedirs(extracted_images)
	extract(original_images, extracted_images)


def detect_features(extracted_images, detected_features):
	"""
	Uses a slightly modified version of Dmitry Poliyivets's FeatureDetector
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
