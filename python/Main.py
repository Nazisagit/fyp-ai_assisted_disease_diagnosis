# Filename: Main.py
# Author: Dmytro Poliyivets
# Institution: King's College London
# Copyright: 2018, Dmytro Poliyivets, King's College London


# Import necessary classes:
from python.IPCLDiagnosis import IPCLDiagnosis
from python.ImageExtractor import ImageExtractor
from python.FeatureDetector import FeatureDetector
from processing.DataProcessing import DataProcessing, DataCollecting
import os

# TODO
# Diagnosis is not efficient enough and takes too long


# Application main method
def main():
    # Instantiate necessary classes:

    type1 = 'type1/'
    type2 = 'type2/'
    type3 = 'type3/'
    type4 = 'type4/'
    type5 = 'type5/'
    patient_number = '0010092102d'
    patient_date = '2017-12-08'
    original_images = '../Student Data/' + patient_number + '/' + patient_date + '/'
    extracted_images = '../extracted_images/' + patient_number + '/' + patient_date + '/'
    detected_features = '../detected_features/' + patient_number + '/' + patient_date + '/'
    data_output = '../data_output/'

    extract_images(original_images, extracted_images)
    feature_tables = detect_features(extracted_images, detected_features)
    #
    # data_collecting = DataCollecting(feature_tables, data_output, type5)
    # data_collecting.init_output_files()
    # data_collecting.save_data()

    # process_data(data_output, type5)

    ipcl_diagnosis = IPCLDiagnosis(feature_tables, detected_features)
    ipcl_diagnosis.analyse_feature_tables()
    ipcl_diagnosis.diagnose_by_type()

    # Instantiate Disease Diagnoser
    # Get feature table resulting from performing feature detection
    # featureTable = featureDetector.get_feature_table()
    # Get the list that contains the number of IPCLs extracted
    # per each frame with feature detection
    # numberIPCLs = featureDetector.get_number_ipcls()
    # Instantiate Diagnoser class
    # diseaseDiagnoser = Diagnoser(featureTable, numberIPCLs)
    # Step 3: Perform Disease Diagnosis
    # diseaseDiagnoser.diagnose()
    # diseaseDiagnoser.naiveBayes()

    # Instantiate Statistical Analysis Class (if needed)
    # Define the folder that contains the frames for
    # which statistical analysis has to be performed
    # in the form: './path_to_your_folder/'
    # inputFolder = './temp/frames/'
    # statisticsAnalyser = StatisticalAnalysis(original_images)
    # 4. Perform Statistics analysis (if needed)
    # statisticsAnalyser.analyse()


def extract_images(original_images, extracted_images):
    if not os.path.exists(extracted_images):
        os.makedirs(extracted_images)
    #
    image_extractor = ImageExtractor(original_images, extracted_images)
    image_extractor.extract()


def detect_features(extracted_images, detected_features):
    if not os.path.exists(detected_features):
        os.makedirs(detected_features)
    # # # Instantiate Feature Detector class
    feature_detector = FeatureDetector(extracted_images, detected_features)
    # # # Step 2: Perform Feature Detection
    feature_detector.run()
    return feature_detector.get_feature_tables()


def process_data(data_output, ipcl_type):
    path = data_output + ipcl_type
    data_processing = DataProcessing()
    for file_name in os.listdir(path):
        if file_name == 'rotation.csv':
            print('\nMean rotation: ' + str(data_processing.calculate_mean(path + file_name)))
            print('Median rotation: ' + str(data_processing.calculate_median(path + file_name)))
            print('Std rotation: ' + str(data_processing.calculate_std(path + file_name)))
            print('Mode rotation: ' + str(data_processing.calculate_modes(path + file_name)))
        elif file_name == 'area.csv':
            print('\nMean area: ' + str(data_processing.calculate_mean(path + file_name)))
            print('Median area: ' + str(data_processing.calculate_median(path + file_name)))
            print('Std area: ' + str(data_processing.calculate_std(path + file_name)))
            print('Mode area: ' + str(data_processing.calculate_modes(path + file_name)))
        elif file_name == 'length.csv':
            print('\nMean length: ' + str(data_processing.calculate_mean(path + file_name)))
            print('Median length: ' + str(data_processing.calculate_median(path + file_name)))
            print('Std length: ' + str(data_processing.calculate_std(path + file_name)))
            print('Mode length: ' + str(data_processing.calculate_modes(path + file_name)))
        elif file_name == 'width.csv':
            print('\nMean width: ' + str(data_processing.calculate_mean(path + file_name)))
            print('Median width: ' + str(data_processing.calculate_median(path + file_name)))
            print('Std width: ' + str(data_processing.calculate_std(path + file_name)))
            print('Mode width:' + str(data_processing.calculate_modes(path + file_name)))
        elif file_name == 'height.csv':
            print('\nMean height: ' + str(data_processing.calculate_mean(path + file_name)))
            print('Median height: ' + str(data_processing.calculate_median(path + file_name)))
            print('Std height: ' + str(data_processing.calculate_std(path + file_name)))
            print('Mode height: ' + str(data_processing.calculate_modes(path + file_name)))


# Call main function
if __name__ == "__main__":
    main()

