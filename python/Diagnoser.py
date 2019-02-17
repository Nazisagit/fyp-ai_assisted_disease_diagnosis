# Filename: Diagnoser.py
# Author: Dmytro Poliyivets
# Institution: King's College London
# Copyright: 2018, Dmytro Poliyivets, King's College London


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
    def __init__(self, featureTable, numberIPCLs):
        # Define an instance of the feature table that stores
        # the features of the extracted IPCLs data
        self.featureTable = featureTable
        # Define a list that contains number of IPCLs per each frame
        self.numberIPCLs = numberIPCLs
        # Define a dictionary that will store the results of statistical analysis of the feature table
        self.statistics = {}

    # Determines diagnose based on the similarity of features to the
    # healthy/unhealthy statistical models. Similarity is measured
    # by absolute distance and percent deviation from the model.
    def diagnose(self):
        # Define statistically determined healthy and unhealthy values
        # (Only strong features are used, i.e. features that have more
        # than 5% deviation between healthy and unhealthy values)
        # Structure [healthy, unhealthy]
        # Mean values
        meanArea = [27.2594414602, 30.0223433195]
        meanOccurrence = [272.298507463, 728.147435897]
        # Median values
        medianArea = [19.5, 23.0]
        medianOccurrence = [201.0, 695.0]
        # Mode values
        # May add mode length
        modeGreen = [133.0, 112.0]
        # Standard Derivation values
        sdArea = [29.0775801293, 26.2012631651]
        sdRed = [22.17655425416077, 24.389095867138533]
        sdLength = [14.0724292282, 11.5186645705]
        sdOccurrence = [269.602112293, 406.893497129]

        # Perform statistical analysis of the current state of the feature table
        self.analyse_feature_table()

        # Perform diagnosis
        # True - means healthy; False - means unhealthy
        # Put all features analysed for diagnosis in a single list
        diagnosedFeatures = []
        diagnosedFeatures.append(self.determine_diagnose(self.statistics['Mean Area'], meanArea[0], meanArea[1]))
        diagnosedFeatures.append(self.determine_diagnose(self.statistics['Mean Occurrence'], meanOccurrence[0], meanOccurrence[1]))
        diagnosedFeatures.append(self.determine_diagnose(self.statistics['Median Area'], medianArea[0], medianArea[1]))
        diagnosedFeatures.append(self.determine_diagnose(self.statistics['Median Occurrence'], medianOccurrence[0], medianOccurrence[1]))
        diagnosedFeatures.append(self.determine_diagnose(self.statistics['SD Area'], sdArea[0], sdArea[1]))
        diagnosedFeatures.append(self.determine_diagnose(self.statistics['SD Colour'][0], sdRed[0], sdRed[1]))
        diagnosedFeatures.append(self.determine_diagnose(self.statistics['SD Length'], sdLength[0], sdLength[1]))
        diagnosedFeatures.append(self.determine_diagnose(self.statistics['SD Occurrence'], sdOccurrence[0], sdOccurrence[1]))

        # In the case multiple modes are present, flatten the results
        dMG = self.flatten([self.determine_diagnose(value, modeGreen[0], modeGreen[1]) for value in self.statistics['Mode Colour'][1]])
        # Find mode of the flattened lists, so that results are not affected by a lot of mode diagnoses
        dMG = self.get_mode_of_lists(dMG)
        # Append mode features diagnosis to the diagnosedFeatures list, checking if multiple modes are present
        if dMG[0] == False or dMG[0] == True:
            diagnosedFeatures.append(dMG)
        else:
            [diagnosedFeatures.append(x) for x in dMG]

        # Eliminate all None results from the diagnosed features list. Here, None results
        # indicate not strong enough diagnosis, so should be neglected
        diagnosedFeatures = [x for x in diagnosedFeatures if x is not None]

        # Separate positive (unhealthy) and negative (healthy) results into two separate lists
        healthy = []  # [True, True]
        unhealthy = []  # [False, False]
        # Loop all values inside diagnosedFeatures
        for value in diagnosedFeatures:
            # If element corresponds to healthy
            if value[0] == True:
                healthy.append(value)
            else:
                unhealthy.append(value)

        # Determine disease probabilities...
        # Determine probability of having the disease
        haveDisease = len(unhealthy) / len(diagnosedFeatures) * 100
        # Determine probability of not having the disease
        notHaveDisease = len(healthy) / len(diagnosedFeatures) * 100

        # Perform actual diagnosis
        if haveDisease > notHaveDisease:
            print('(Similarity-Based). I am sorry, you have Acid-Reflux. Probability:', haveDisease)
            print()
        else:
            print('(Similarity-Based). Congratulations, you are healthy. Probability:', notHaveDisease)
            print()

    # Determines strong features that can be used for diagnosis
    # by calculating ratios and absolute distances
    def determine_diagnose(self, result, healthy, unhealthy):
        # Calculate ratio
        ratioHealthy = abs(result) / abs(healthy)
        ratioUnHealthy = abs(result) / abs(unhealthy)

        # Perform normalization
        normHealthy = self.normalize(ratioHealthy)
        normUnHealthy = self.normalize(ratioUnHealthy)

        # Determine diagnose for this feature
        if abs(normHealthy) < abs(normUnHealthy):
            # Ratio corresponds to ratio of healthy person
            # Now this has to be confirmed by absolute distance
            absDistHealthy = abs(abs(healthy) - abs(result))
            absDistUnHealthy = abs(abs(unhealthy) - abs(result))

            # Return the results
            if absDistHealthy < absDistUnHealthy:
                return [True, True]
            else:
                # Return None because result is not so reliable
                return None
        else:
            # Ratio corresponds to ratio of unhealthy person
            # Now this has to be confirmed by absolute distance
            absDistHealthy = abs(abs(healthy) - abs(result))
            absDistUnHealthy = abs(abs(unhealthy) - abs(result))

            # Return the results
            if absDistUnHealthy < absDistHealthy:
                return [False, False]
            else:
                # Return None because result is not so reliable
                return None

    # Calculates the statistical analysis of the feature table
    def analyse_feature_table(self):
        # Define vectors necessary for calculation
        rotationVector = []
        areaVector = []
        redVector = []
        greenVector = []
        blueVector = []
        lengthVector = []

        # Get height of the feature table
        tableHeight = len(self.featureTable)
        # For Rotation, Area, Colour, and Length
        for feature in [3, 4, 5, 6]:
            # For all entries in the table
            for row in range(0, tableHeight):
                # Calculate Total Rotation
                if feature == 3:
                    rotationVector.append(self.featureTable[row][feature])
                # Calculate Total Area
                elif feature == 4:
                    areaVector.append(self.featureTable[row][feature])
                # Calculate Total Colour
                elif feature == 5:
                    redVector.append(self.featureTable[row][feature][0])
                    greenVector.append(self.featureTable[row][feature][1])
                    blueVector.append(self.featureTable[row][feature][2])
                # Calculate Total Length
                elif feature == 6:
                    lengthVector.append(self.featureTable[row][feature])

        # Calculate statistics
        # Calculate mean
        meanRotation = np.mean(rotationVector)
        meanArea = np.mean(areaVector)
        meanColour = (np.mean(redVector), np.mean(greenVector), np.mean(blueVector))
        meanLength = np.mean(lengthVector)
        meanOccurrence = np.mean(self.numberIPCLs)
        # Calculate median
        medianRotation = np.median(rotationVector)
        medianArea = np.median(areaVector)
        medianColour = (np.median(redVector), np.median(greenVector), np.median(blueVector))
        medianLength = np.median(lengthVector)
        medianOccurrence = np.median(self.numberIPCLs)
        # Calculate Standard Derivation
        sdRotation = np.std(rotationVector, dtype=np.float64)
        sdArea = np.std(areaVector, dtype=np.float64)
        sdColour = (np.std(redVector, dtype=np.float64), np.std(greenVector, dtype=np.float64),
                    np.std(blueVector, dtype=np.float64))
        sdLength = np.std(lengthVector, dtype=np.float64)
        sdOccurrence = np.std(self.numberIPCLs, dtype=np.float64)
        # Calculate mode
        modeRotation = self.get_mode(rotationVector)
        modeArea = self.get_mode(areaVector)
        modeColour = (self.get_mode(redVector), self.get_mode(greenVector), self.get_mode(blueVector))
        modeLength = self.get_mode(lengthVector)
        modeOccurrence = self.get_mode(self.numberIPCLs)

        # Clear current state of the dictionary
        self.statistics.clear()
        # Record the results in the dictionary in the state
        self.statistics['Mean Rotation'] = meanRotation
        self.statistics['Mean Area'] = meanArea
        self.statistics['Mean Colour'] = meanColour
        self.statistics['Mean Length'] = meanLength
        self.statistics['Mean Occurrence'] = meanOccurrence
        self.statistics['Median Rotation'] = medianRotation
        self.statistics['Median Area'] = medianArea
        self.statistics['Median Colour'] = medianColour
        self.statistics['Median Length'] = medianLength
        self.statistics['Median Occurrence'] = medianOccurrence
        self.statistics['SD Rotation'] = sdRotation
        self.statistics['SD Area'] = sdArea
        self.statistics['SD Colour'] = sdColour
        self.statistics['SD Length'] = sdLength
        self.statistics['SD Occurrence'] = sdOccurrence
        self.statistics['Mode Rotation'] = modeRotation
        self.statistics['Mode Area'] = modeArea
        self.statistics['Mode Colour'] = modeColour
        self.statistics['Mode Length'] = modeLength
        self.statistics['Mode Occurrence'] = modeOccurrence

        # Print the results to the console
        print('----------------------------------------------------------------------------')
        print('Mean Rotation', meanRotation)
        print('Mean Area', meanArea)
        print('Mean Colour', meanColour)
        print('Mean Length', meanLength)
        print('Mean Occurrences', meanOccurrence)
        print('----------------------------------------------------------------------------')
        print('Median Rotation', medianRotation)
        print('Median Area', medianArea)
        print('Median Colour', medianColour)
        print('Median Length', medianLength)
        print('Median Occurrences', medianOccurrence)
        print('----------------------------------------------------------------------------')
        print('Std Rotation', sdRotation)
        print('Std Area', sdArea)
        print('Std Colour', sdColour)
        print('Std Length', sdLength)
        print('Std Occurrences', sdOccurrence)
        print('----------------------------------------------------------------------------')
        print('Mode Rotation', modeRotation)
        print('Mode Area', modeArea)
        print('Mode Colour', modeColour)
        print('Mode Length', modeLength)
        print('Mode Occurrence', modeOccurrence)
        print('----------------------------------------------------------------------------')
        print('Elements:', tableHeight)
        print()

    # Performs normalization of the argument
    def normalize(self, number):
        return (number*100)-100

    # Flattens the list. Used to diagnose mode features when multiple modes are present
    def flatten(self, list):
        # Define an output list
        output = []

        # Loop all values
        for value in list:
            if value is not None:
                if value[0] == True or value[0] == False:
                    output.append(value)
                else:
                    for element in value:
                        output.append(element)

        return output

    # Performs multiple mode estimation
    ########## 29/01/2019
    # Added a check to see if there where any most common items
    # Else, return an empty array
    def get_mode(self, list):
        counter = Counter(list)
        if len(counter.most_common(1)) >= 1:
            _, val = counter.most_common(1)[0]
            return [x for x, y in counter.items() if y == val]
        else:
            return []

    # Performs multiple mode estimation for diagnosing purposes (using lists)
    def get_mode_of_lists(self, list):
        f = list.count([False, False])
        t = list.count([True, True])

        if f > t:
            return [False, False]
        elif t > f:
            return [True, True]
        elif t == f:
            return [[False, False], [True, True]]


    # Naive Bayes Classifier code starts here...
    # Determines diagnose based on the Naive Bayes compound probability.
    def naiveBayes(self):
        # Define statistically determined healthy and unhealthy values
        # Structure [healthy, unhealthy]
        # Area
        meanArea = [27.2594414602, 30.0223433195]
        sdArea = [29.0775801293, 26.2012631651]
        # Blue colour
        meanBlue = [95.4698320914912, 98.6696582670827]
        sdBlue = [20.0228196493808, 20.8765578674837]
        # Occurrence
        meanOccurrence = [272.298507463, 728.147435897]
        sdOccurrence = [29.0775801293, 26.2012631651]
        # Length
        meanLength = [15.9788971717, 15.552534972]
        sdLength = [14.0724292282, 11.5186645705]
        # Red colour
        meanRed = [123.538690343856, 125.037807140195]
        sdRed = [22.1765542541607, 24.3890958671385]
        # Green colour
        meanGreen = [115.816017805436, 118.552651471751]
        sdGreen = [22.5216184323126, 22.7929854656791]

        # Perform statistical analysis of the current state of the feature table
        self.analyse_feature_table()

        # Define a dictionary storing the results
        diagnoses = {}
        diagnoses['Healthy'] = 1
        diagnoses['Unhealthy'] = 1

        # Calculate the probability of data belonging to healthy class
        diagnoses['Healthy'] *= self.calculateProbability(self.statistics['Mean Area'], meanArea[0], sdArea[0])
        diagnoses['Healthy'] *= self.calculateProbability(self.statistics['Mean Colour'][2], meanBlue[0], sdBlue[0])
        diagnoses['Healthy'] *= self.calculateProbability(self.statistics['Mean Colour'][1], meanGreen[0], sdGreen[0])
        diagnoses['Healthy'] *= self.calculateProbability(self.statistics['Mean Colour'][0], meanRed[0], sdRed[0])
        diagnoses['Healthy'] *= self.calculateProbability(self.statistics['Mean Occurrence'], meanOccurrence[0], sdOccurrence[0])
        diagnoses['Healthy'] *= self.calculateProbability(self.statistics['Mean Length'], meanLength[0], sdLength[0])

        # Calculate the probability of data belonging to healthy class
        diagnoses['Unhealthy'] *= self.calculateProbability(self.statistics['Mean Area'], meanArea[1], sdArea[1])
        diagnoses['Unhealthy'] *= self.calculateProbability(self.statistics['Mean Colour'][2], meanBlue[1], sdBlue[1])
        diagnoses['Unhealthy'] *= self.calculateProbability(self.statistics['Mean Colour'][1], meanGreen[1], sdGreen[1])
        diagnoses['Unhealthy'] *= self.calculateProbability(self.statistics['Mean Colour'][0], meanRed[1], sdRed[1])
        diagnoses['Unhealthy'] *= self.calculateProbability(self.statistics['Mean Occurrence'], meanOccurrence[1], sdOccurrence[1])
        diagnoses['Unhealthy'] *= self.calculateProbability(self.statistics['Mean Length'], meanLength[1], sdLength[1])

        # Normalisation constant to identify class probability
        normalisationConstant = diagnoses['Healthy'] + diagnoses['Unhealthy']

        # Select maximum of the two naive Bayes probabilities
        if diagnoses['Healthy'] > diagnoses['Unhealthy']:
            print('(Naive Bayes). Congratulations, you are healthy. Probability:', diagnoses['Healthy'] / normalisationConstant * 100)
        else:
            print('(Naive Bayes). I am sorry, you have Acid-Reflux. Probability:', diagnoses['Unhealthy'] / normalisationConstant * 100)

    # Calculates conditional class probability
    def calculateProbability(self, data, mean, stDeviation):
        # Calculate probability of belonging to the class
        exponent = math.exp(-(math.pow(data - mean, 2) / (2 * math.pow(stDeviation, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stDeviation)) * exponent