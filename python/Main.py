# Filename: Main.py
# Author: Dmytro Poliyivets
# Institution: King's College London
# Copyright: 2018, Dmytro Poliyivets, King's College London


# Import necessary classes:
from python.IPCLDiagnosis import IPCLDiagnosis
from python.ImageExtractor import ImageExtractor
from python.FeatureDetector import FeatureDetector
from processing.DataProcessing import DataProcessing
import os

# TODO
# Diagnosis is not efficient enough and takes too long


# Application main method
def main():
    # Instantiate necessary classes:

    patient_number = '0014057701d'
    patient_date = '2017-09-01'
    original_images = '../Student Data/' + patient_number + '/' + patient_date + '/'
    extracted_images = '../extracted_images/' + patient_number + '/' + patient_date + '/'

    if not os.path.exists(extracted_images):
        os.makedirs(extracted_images)

    image_extractor = ImageExtractor(original_images, extracted_images)
    image_extractor.extract()

    detected_features = '../detected_features/' + patient_number + '/' + patient_date + '/'
    if not os.path.exists(detected_features):
        os.makedirs(detected_features)
    # Instantiate Feature Detector class
    feature_detector = FeatureDetector(extracted_images, detected_features)
    # Step 2: Perform Feature Detection
    feature_detector.run()

    feature_detector.save_feature_table()

    feature_tables = feature_detector.get_feature_tables()

    ipcl_diagnosis = IPCLDiagnosis(feature_tables, detected_features)
    ipcl_diagnosis.analyse_feature_tables()
    ipcl_diagnosis.diagnose_by_type()

    data_output = '../data_output/'
    data_processing = DataProcessing(feature_tables, data_output)
    data_processing.get_data()


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

# Call main function
if __name__ == "__main__":
    main()
