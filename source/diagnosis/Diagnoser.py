# Filename: Diagnoser.py
# Author: Dmytro Poliyivets
# Institution: King's College London
# Copyright: 2018, Dmytro Poliyivets, King's College London
# Last modified: 05/04/2019

"""
This module has been modified to pythonize it's style (e.g. function names,
variable names).
"""

# Import statements
import math
import numpy as np
from collections import Counter


# The class is intended for diagnosing Acid-Reflux based on the
# IPCL features extracted from the endoscopic footage and the
# healthy/unhealthy statistical model. Two different implementations are used:
# 1. Diagnosing by similarity (absolute distance and percent deviation)
# 2. Naive Bayes probability estimation.
class Diagnoser:

    # Class constructor
    def __init__(self, feature_tables):
        # Define an instance of the feature table that stores
        # the features of the extracted IPCLs data
        self.feature_tables = feature_tables
        # Define a dictionary that will store the results of statistical analysis of the feature table
        self.statistics = dict()

    # Calculates the statistical analysis of the feature table
    def analyse_feature_table(self):
        # Define vectors necessary for calculation
        # Modified
        # Removed rotatationVector
        # Renamed the *Vector to *_list
        width_list = list()
        height_list = list()
        area_list = list()
        red_list = list()
        green_list = list()
        blue_vector = list()
        length_list = list()

        for table in self.feature_tables:
            # Get height of the feature table
            table_height = len(table)
            # For Width, Height, Area, Colour, and Length
            for feature in [0, 1, 2, 3, 4]:
                # For all entries in the table
                for row in range(0, table_height):
                    if feature == 0:
                        width_list.append(table[row][feature])
                    elif feature == 1:
                        height_list.append(table[row][feature])
                    elif feature == 2:
                        area_list.append(table[row][feature])
                    elif feature == 3:
                        red_list.append(table[row][feature][0])
                        green_list.append(table[row][feature][1])
                        blue_vector.append(table[row][feature][2])
                    elif feature == 4:
                        length_list.append(table[row][feature])

        # Calculate statistics
        # Calculate mean
        mean_width = np.mean(width_list)
        mean_height = np.mean(height_list)
        mean_area = np.mean(area_list)
        mean_colour = (np.mean(red_list), np.mean(green_list), np.mean(blue_vector))
        mean_length = np.mean(length_list)
        # Calculate median
        median_width = np.median(width_list)
        median_height = np.median(height_list)
        median_area = np.median(area_list)
        median_colour = (np.median(red_list), np.median(green_list), np.median(blue_vector))
        median_length = np.median(length_list)
        # Calculate Standard Derivation
        sd_width = np.std(width_list, dtype=np.float64)
        sd_height = np.std(height_list, dtype=np.float64)
        sd_area = np.std(area_list, dtype=np.float64)
        sd_colour = (np.std(red_list, dtype=np.float64), np.std(green_list, dtype=np.float64),
                    np.std(blue_vector, dtype=np.float64))
        sd_length = np.std(length_list, dtype=np.float64)
        # Calculate mode
        mode_width = self.get_mode(width_list)
        mode_height = self.get_mode(height_list)
        mode_area = self.get_mode(area_list)
        mode_colour = (self.get_mode(red_list), self.get_mode(green_list), self.get_mode(blue_vector))
        mode_length = self.get_mode(length_list)

        # Clear current state of the dictionary
        self.statistics.clear()
        # Record the results in the dictionary in the state
        self.statistics['Mean Width'] = mean_width
        self.statistics['Mean Height'] = mean_height
        self.statistics['Mean Area'] = mean_area
        self.statistics['Mean Colour'] = mean_colour
        self.statistics['Mean Length'] = mean_length
        self.statistics['Median Width'] = median_width
        self.statistics['Median Height'] = median_height
        self.statistics['Median Area'] = median_area
        self.statistics['Median Colour'] = median_colour
        self.statistics['Median Length'] = median_length
        self.statistics['SD Width'] = sd_width
        self.statistics['SD Height'] = sd_height
        self.statistics['SD Area'] = sd_area
        self.statistics['SD Colour'] = sd_colour
        self.statistics['SD Length'] = sd_length
        self.statistics['Mode Width'] = mode_width
        self.statistics['Mode Height'] = mode_height
        self.statistics['Mode Area'] = mode_area
        self.statistics['Mode Colour'] = mode_colour
        self.statistics['Mode Length'] = mode_length

        # Print the results to the console
        print('----------------------------------------------------------------------------')
        print('Mean Width', mean_width)
        print('Mean Height', mean_height)
        print('Mean Area', mean_area)
        print('Mean Colour', mean_colour)
        print('Mean Length', mean_length)
        print('----------------------------------------------------------------------------')
        print('Median Width', median_width)
        print('Median Height', median_height)
        print('Median Area', median_area)
        print('Median Colour', median_colour)
        print('Median Length', median_length)
        print('----------------------------------------------------------------------------')
        print('Std Width', sd_width)
        print('Std Height', sd_height)
        print('Std Area', sd_area)
        print('Std Colour', sd_colour)
        print('Std Length', sd_length)
        print('----------------------------------------------------------------------------')
        print('Mode Width', mode_width)
        print('Mode Height', mode_height)
        print('Mode Area', mode_area)
        print('Mode Colour', mode_colour)
        print('Mode Length', mode_length)
        print('----------------------------------------------------------------------------')
        print()

    # Performs multiple mode estimation
    # Modified
    # changed into a static method
    @staticmethod
    def get_mode(_list):
        counter = Counter(_list)
        _, val = counter.most_common(1)[0]
        return [x for x, y in counter.items() if y == val]

    # Naive Bayes Classifier code starts here...
    # Determines diagnose based on the Naive Bayes compound probability.
    def naive_bayes(self):
        # Define statistically determined healthy and unhealthy values
        # Modified
        # features used in the diagnosis are
        # Width, Height, Area, Colour
        # Structure [Group 1, Group 2, Group 3]
        # Width
        mean_width = [8.239755, 7.752341, 7.604807]
        sd_width = [7.272695, 6.07759, 6.523073]
        # Height
        mean_height = [8.181128, 7.811186, 7.574423]
        sd_height = [7.088363, 5.9439, 6.465264]
        # Area
        mean_area = [23.681036, 20.505368, 22.695575]
        sd_area = [25.966122, 21.795541, 26.583639]
        # Red colour
        mean_red = [98.276078, 109.314981, 102.045455]
        sd_red = [34.581402, 24.362770, 25.589054]
        # Green colour
        mean_green = [74.935504, 83.073079, 77.477525]
        sd_green = [24.786052, 22.269277, 23.932506]
        # Blue colour
        mean_blue = [95.374316, 105.005287, 98.653857]
        sd_blue = [28.716705, 24.780164, 26.931724]
        # Length
        mean_length = [19.378218, 18.711602, 17.63317]
        sd_length = [17.262154, 15.360977, 16.3681]


        # Perform statistical analysis of the current state of the feature table
        self.analyse_feature_table()

        # Define a dictionary storing the results
        diagnoses = dict()
        diagnoses['Group 1'] = 1
        diagnoses['Group 2'] = 1
        diagnoses['Group 3'] = 1

        diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Width'], mean_width[0], sd_width[0])
        diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Height'], mean_height[0], sd_height[0])
        diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Area'], mean_area[0], sd_area[0])
        diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Colour'][0], mean_red[0], sd_red[0])
        diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Colour'][1], mean_green[0], sd_green[0])
        diagnoses['Group 1'] *= self.calculate_probability(self.statistics['Mean Colour'][2], mean_blue[0], sd_blue[0])
        # diagnoses['Group 1'] *= self.calculateProbability(self.statistics['Mean Length'], mean_length[0], sd_length[0])

        diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Width'], mean_width[1], sd_width[1])
        diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Height'], mean_height[1], sd_height[1])
        diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Area'], mean_area[1], sd_area[1])
        diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Colour'][0], mean_red[1], sd_red[1])
        diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Colour'][1], mean_green[1], sd_green[1])
        diagnoses['Group 2'] *= self.calculate_probability(self.statistics['Mean Colour'][2], mean_blue[1], sd_blue[1])
        # diagnoses['Group 2'] *= self.calculateProbability(self.statistics['Mean Length'], mean_length[1], sd_length[1])

        diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Width'], mean_width[2], sd_width[2])
        diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Height'], mean_height[2], sd_height[2])
        diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Area'], mean_area[2], sd_area[2])
        diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Colour'][0], mean_red[2], sd_red[2])
        diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Colour'][1], mean_green[2], sd_green[2])
        diagnoses['Group 3'] *= self.calculate_probability(self.statistics['Mean Colour'][2], mean_blue[2], sd_blue[2])
        # diagnoses['Group 3'] *= self.calculateProbability(self.statistics['Mean Length'], mean_length[2], sd_length[2])

        # Normalisation constant to identify class probability
        normalisation_constant = diagnoses['Group 1'] + diagnoses['Group 2'] + diagnoses['Group 3']

        # Select maximum of the two naive Bayes probabilities
        if diagnoses['Group 1'] > diagnoses['Group 2'] and diagnoses['Group 1'] > diagnoses['Group 3']:
            print('Group 1 most likely. (Non-neoplastic)')
            print('(Naive Bayes). Group 1. Probability:', diagnoses['Group 1'] / normalisation_constant * 100)
            print('(Naive Bayes). Group 2. Probability:', diagnoses['Group 2'] / normalisation_constant * 100)
            print('(Naive Bayes). Group 3. Probability:', diagnoses['Group 3'] / normalisation_constant * 100)
        elif diagnoses['Group 2'] > diagnoses['Group 1'] and diagnoses['Group 2'] > diagnoses['Group 3']:
            print('Group 2 most likely. (Borderline)')
            print('(Naive Bayes). Group 2. Probability:', diagnoses['Group 2'] / normalisation_constant * 100)
            print('(Naive Bayes). Group 1. Probability:', diagnoses['Group 1'] / normalisation_constant * 100)
            print('(Naive Bayes). Group 3. Probability:', diagnoses['Group 3'] / normalisation_constant * 100)
        elif diagnoses['Group 3'] > diagnoses['Group 1'] and diagnoses['Group 3'] > diagnoses['Group 2']:
            print('Group 3 most likely. (Cancer)')
            print('(Naive Bayes). Group 3. Probability:', diagnoses['Group 3'] / normalisation_constant * 100)
            print('(Naive Bayes). Group 1. Probability:', diagnoses['Group 1'] / normalisation_constant * 100)
            print('(Naive Bayes). Group 2. Probability:', diagnoses['Group 2'] / normalisation_constant * 100)

    # Calculates conditional class probability
    # Modified
    # changed into a static method
    @staticmethod
    def calculate_probability(data, mean, sd):
        # Calculate probability of belonging to the class
        exponent = math.exp(-(math.pow(data - mean, 2) / (2 * math.pow(sd, 2))))
        return (1 / (math.sqrt(2 * math.pi) * sd)) * exponent
