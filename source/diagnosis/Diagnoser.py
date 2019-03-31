# Filename: Diagnoser.py
# Author: Dmytro Poliyivets
# Institution: King's College London
# Copyright: 2018, Dmytro Poliyivets, King's College London
# Last modified: 30/03/2019


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
    def __init__(self, featureTables):
        # Define an instance of the feature table that stores
        # the features of the extracted IPCLs data
        self.featureTables = featureTables
        # Define a dictionary that will store the results of statistical analysis of the feature table
        self.statistics = {}

    # Calculates the statistical analysis of the feature table
    def analyse_feature_table(self):
        # Define vectors necessary for calculation
        widthVector = []
        heightVector = []
        areaVector = []
        redVector = []
        greenVector = []
        blueVector = []
        lengthVector = []

        for table in self.featureTables:
            # Get height of the feature table
            tableHeight = len(table)
            # For Width, Height, Area, Colour, and Length
            for feature in [0, 1, 2, 3, 4]:
                # For all entries in the table
                for row in range(0, tableHeight):
                    if feature == 0:
                        widthVector.append(table[row][feature])
                    elif feature == 1:
                        heightVector.append(table[row][feature])
                    elif feature == 2:
                        areaVector.append(table[row][feature])
                    elif feature == 3:
                        redVector.append(table[row][feature][0])
                        greenVector.append(table[row][feature][1])
                        blueVector.append(table[row][feature][2])
                    elif feature == 4:
                        lengthVector.append(table[row][feature])

        # Calculate statistics
        # Calculate mean
        meanWidth = np.mean(widthVector)
        meanHeight = np.mean(heightVector)
        meanArea = np.mean(areaVector)
        meanColour = (np.mean(redVector), np.mean(greenVector), np.mean(blueVector))
        meanLength = np.mean(lengthVector)
        # Calculate median
        medianWidth = np.median(widthVector)
        medianHeight = np.median(heightVector)
        medianArea = np.median(areaVector)
        medianColour = (np.median(redVector), np.median(greenVector), np.median(blueVector))
        medianLength = np.median(lengthVector)
        # Calculate Standard Derivation
        sdWidth = np.std(widthVector, dtype=np.float64)
        sdHeight = np.std(heightVector, dtype=np.float64)
        sdArea = np.std(areaVector, dtype=np.float64)
        sdColour = (np.std(redVector, dtype=np.float64), np.std(greenVector, dtype=np.float64),
                    np.std(blueVector, dtype=np.float64))
        sdLength = np.std(lengthVector, dtype=np.float64)
        # Calculate mode
        modeWidth = self.get_mode(widthVector)
        modeHeight = self.get_mode(heightVector)
        modeArea = self.get_mode(areaVector)
        modeColour = (self.get_mode(redVector), self.get_mode(greenVector), self.get_mode(blueVector))
        modeLength = self.get_mode(lengthVector)

        # Clear current state of the dictionary
        self.statistics.clear()
        # Record the results in the dictionary in the state
        self.statistics['Mean Width'] = meanWidth
        self.statistics['Mean Height'] = meanHeight
        self.statistics['Mean Area'] = meanArea
        self.statistics['Mean Colour'] = meanColour
        self.statistics['Mean Length'] = meanLength
        self.statistics['Median Width'] = medianWidth
        self.statistics['Median Height'] = medianHeight
        self.statistics['Median Area'] = medianArea
        self.statistics['Median Colour'] = medianColour
        self.statistics['Median Length'] = medianLength
        self.statistics['SD Width'] = sdWidth
        self.statistics['SD Height'] = sdHeight
        self.statistics['SD Area'] = sdArea
        self.statistics['SD Colour'] = sdColour
        self.statistics['SD Length'] = sdLength
        self.statistics['Mode Width'] = modeWidth
        self.statistics['Mode Height'] = modeHeight
        self.statistics['Mode Area'] = modeArea
        self.statistics['Mode Colour'] = modeColour
        self.statistics['Mode Length'] = modeLength

        # Print the results to the console
        print('----------------------------------------------------------------------------')
        print('Mean Width', meanWidth)
        print('Mean Height', meanHeight)
        print('Mean Area', meanArea)
        print('Mean Colour', meanColour)
        print('Mean Length', meanLength)
        print('----------------------------------------------------------------------------')
        print('Median Width', medianWidth)
        print('Median Height', medianHeight)
        print('Median Area', medianArea)
        print('Median Colour', medianColour)
        print('Median Length', medianLength)
        print('----------------------------------------------------------------------------')
        print('Std Width', sdWidth)
        print('Std Height', sdHeight)
        print('Std Area', sdArea)
        print('Std Colour', sdColour)
        print('Std Length', sdLength)
        print('----------------------------------------------------------------------------')
        print('Mode Width', modeWidth)
        print('Mode Height', modeHeight)
        print('Mode Area', modeArea)
        print('Mode Colour', modeColour)
        print('Mode Length', modeLength)
        print('----------------------------------------------------------------------------')
        print()

    # Performs multiple mode estimation
    def get_mode(self, list):
        counter = Counter(list)
        _, val = counter.most_common(1)[0]
        return [x for x, y in counter.items() if y == val]

    # Naive Bayes Classifier code starts here...
    # Determines diagnose based on the Naive Bayes compound probability.
    def naiveBayes(self):
        # Define statistically determined healthy and unhealthy values
        # Structure [Group 1, Group 2, Group 3]
        # Width
        meanWidth = [8.289956, 7.729065, 9.025486]
        sdWidth = [7.614713, 5.813382, 8.720085]
        #Height
        meanHeight = [8.331788, 7.814836, 8.417027]
        sdHeight = [7.38072, 5.735939, 8.688936]
        # Area
        meanArea = [24.361315, 20.95668, 28.510092]
        sdArea = [32.265274, 21.614612, 57.940803]
        # Red colour
        meanRed = [98.991626, 108.469259, 114.401183]
        sdRed = [34.754349, 23.072570, 25.851544]
        # Green colour
        meanGreen = [95.551225, 106.184371, 90.892506]
        sdGreen = [28.591656, 24.205274, 24.575417]
        # Blue colour
        meanBlue = [75.146324, 84.171705, 70.600833]
        sdBlue = [24.648204, 21.767228, 21.425945]
        # Length
        meanLength = [19.590305, 18.56163, 20.86052]
        sdLength = [19.240105, 14.642425, 27.657586]

        # Perform statistical analysis of the current state of the feature table
        self.analyse_feature_table()

        # Define a dictionary storing the results
        diagnoses = {}
        diagnoses['Group 1'] = 1
        diagnoses['Group 2'] = 1
        diagnoses['Group 3'] = 1

        # Calculate the probability of data belonging to healthy class
        diagnoses['Group 1'] *= self.calculateProbability(self.statistics['Mean Width'], meanWidth[0], sdWidth[0])
        diagnoses['Group 1'] *= self.calculateProbability(self.statistics['Mean Colour'][2], meanBlue[0], sdBlue[0])
        diagnoses['Group 1'] *= self.calculateProbability(self.statistics['Mean Colour'][1], meanGreen[0], sdGreen[0])
        diagnoses['Group 1'] *= self.calculateProbability(self.statistics['Mean Colour'][0], meanRed[0], sdRed[0])
        diagnoses['Group 1'] *= self.calculateProbability(self.statistics['Mean Length'], meanLength[0], sdLength[0])

        # Calculate the probability of data belonging to healthy class
        diagnoses['Group 2'] *= self.calculateProbability(self.statistics['Mean Width'], meanWidth[1], sdWidth[1])
        diagnoses['Group 2'] *= self.calculateProbability(self.statistics['Mean Colour'][2], meanBlue[1], sdBlue[1])
        diagnoses['Group 2'] *= self.calculateProbability(self.statistics['Mean Colour'][1], meanGreen[1], sdGreen[1])
        diagnoses['Group 2'] *= self.calculateProbability(self.statistics['Mean Colour'][0], meanRed[1], sdRed[1])
        diagnoses['Group 2'] *= self.calculateProbability(self.statistics['Mean Length'], meanLength[1], sdLength[1])

        diagnoses['Group 3'] *= self.calculateProbability(self.statistics['Mean Width'], meanWidth[2], sdWidth[2])
        diagnoses['Group 3'] *= self.calculateProbability(self.statistics['Mean Colour'][2], meanBlue[2], sdBlue[2])
        diagnoses['Group 3'] *= self.calculateProbability(self.statistics['Mean Colour'][1], meanGreen[2], sdGreen[2])
        diagnoses['Group 3'] *= self.calculateProbability(self.statistics['Mean Colour'][0], meanRed[2], sdRed[2])
        diagnoses['Group 3'] *= self.calculateProbability(self.statistics['Mean Length'], meanLength[2], sdLength[2])

        # Normalisation constant to identify class probability
        normalisationConstant = diagnoses['Group 1'] + diagnoses['Group 2'] + diagnoses['Group 3']

        # Select maximum of the two naive Bayes probabilities
        if diagnoses['Group 1'] > diagnoses['Group 2'] and diagnoses['Group 1'] > diagnoses['Group 3']:
            print('(Naive Bayes). Group 1. Probability:', diagnoses['Group 1'] / normalisationConstant * 100)
        elif diagnoses['Group 2'] > diagnoses['Group 1'] and diagnoses['Group 2'] > diagnoses['Group 3']:
            print('(Naive Bayes). Group 2. Probability:', diagnoses['Group 2'] / normalisationConstant * 100)
        elif diagnoses['Group 3'] > diagnoses['Group 1'] and diagnoses['Group 3'] > diagnoses['Group 2']:
            print('(Naive Bayes). Group 3. Probability:', diagnoses['Group 3'] / normalisationConstant * 100)

    # Calculates conditional class probability
    def calculateProbability(self, data, mean, stDeviation):
        # Calculate probability of belonging to the class
        exponent = math.exp(-(math.pow(data - mean, 2) / (2 * math.pow(stDeviation, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stDeviation)) * exponent