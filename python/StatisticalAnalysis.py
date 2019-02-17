# Filename: StatisticalAnalysis.py
# Author: Dmytro Poliyivets
# Institution: King's College London
# Copyright: 2018, Dmytro Poliyivets, King's College London


# Import statements
from os.path import isfile, join
from collections import Counter
from os import listdir
import numpy as np
import cv2


# Performs statistical analysis of the extracted IPCL features. The functionality includes:
# 1. Mean estimation
# 2. Median estimation
# 3. Mode estimation
# 4. Standard Deviation estimation
# Some code here is duplicated from FeatureDetector.py and Diagnoser.py because it is
# intended to be used as a standalone application that is not dependent on any other
# software packages.
class StatisticalAnalysis:

    # Class constructor
    def __init__(self, inputFolder):
        # Define the folder that contains the frames for
        # which statistical analysis has to be performed
        self.inputFolder = inputFolder
        # Define a table that will store extracted features, with structure:
        # [['(X, Y) Coordinate', 'Width', 'Height', 'Rotation', 'Area', 'Colour (R,G,B)', 'Length']]
        self.featureTable = []
        # Define a list that contains number of IPCLs per each frame
        self.numberIPCLs = []
        # Define a dictionary that will store the results of statistical analysis of the feature table
        self.statistics = {}

    # Runs statistical analysis process
    def analyse(self):
        # Define the files contained inside the input folder
        files = [f for f in listdir(self.inputFolder) if isfile(join(self.inputFolder, f))]

        print('\nAnalysis starts, please wait...\n')

        # For all files in the specified directory
        for frame in files:
            # Check that current file has .png extension
            if frame.endswith('.png'):
                # Load each image
                image = cv2.imread(self.inputFolder + str(frame))

                # Runs feature extraction followed by statistical analysis of that data
                self.statistical_analysis(image)

        # Runs statistical analysis of feature table and print its state
        self.print_state()

    # Perform statistical analysis
    def statistical_analysis(self, frame):
        # Extract features from current image and add it to feature table
        self.extract_features(frame)

        # Uncomment the line below to perform statistical analysis for each frame
        # self.analyseFeatureTable()

    # Identifies the IPCLs structures in the frame and performs
    # feature extraction for each of the identified individual IPCLs.
    def extract_features(self, roi):
        # Mask the back background
        bgMask = cv2.inRange(roi, np.array([0, 0, 0], dtype="uint8"), np.array([0, 0, 0], dtype="uint8"))

        # Dilate the mask a little bit
        bgDilated = cv2.dilate(bgMask, np.ones((5, 5), np.uint8), iterations=1)

        # Threshold the necessary data from the roi
        binaryROI = self.adaptive_threshold_roi(roi)

        # Subtract the dilated roi mask from the threshold
        # image to eliminate border edges
        binaryROI = cv2.subtract(binaryROI, bgDilated)

        # Perform feature extraction for each blob...
        # Identify the contours in the binary roi and copy the roi
        _, contours, _ = cv2.findContours(binaryROI, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        output = roi.copy()
        # Loop all contours
        for c in contours:
            # Check if blob contains glare
            # Get blob dimensions and location
            x, y, w, h = cv2.boundingRect(c)
            # Crop the image containing the blob from ROI
            blobMask = np.zeros(roi.shape[:2], np.uint8)
            cv2.drawContours(blobMask, [c], -1, 255, -1)
            blob = cv2.bitwise_and(roi, roi, mask=blobMask)
            blob = blob[y:y + h, x:x + w]

            # Make sure blob has no glare
            if not self.is_glare(blob):
                # Create a mask containing this blob
                mask = np.zeros(binaryROI.shape, np.uint8)
                cv2.drawContours(mask, [c], 0, 255, -1)

                # Get the mean BGR colour of the blob
                blue, green, red, _ = cv2.mean(roi, mask=mask)

                # Record the mean colour
                meanColour = (red, green, blue)

                # Get the bounding rectangle around the blob, taking into
                # account its area and rotation
                rect = cv2.minAreaRect(c)

                # Get the area of the current blob
                area = cv2.contourArea(c)

                # Get the length of the contour
                length = float(self.line_length(mask))

                # Create new table row entry
                # [['(X, Y) Coordinate', 'Width', 'Height', 'Rotation', 'Area', 'Colour (R,G,B)', 'Length']]
                row = [rect[0], rect[1][0], rect[1][1], rect[2], area, meanColour, length]
                # Add that new row to the feature table
                self.featureTable.append(row)

                # Get the bounding box and draw it on the roi copy
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(output, [box], 0, (0, 0, 255), 2)

        # Record the number of IPCLs per current frame in the state (numberIPCLs)
        self.numberIPCLs.append(len(contours))

        # Print the state of the feature table
        self.print_feature_table()

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
        sdColour = (np.std(redVector, dtype=np.float64), np.std(greenVector, dtype=np.float64), np.std(blueVector, dtype=np.float64))
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

    # Performs adaptive threshold on the ROI to identify vein structures
    def adaptive_threshold_roi(self, frame):
        # Convert the frame to greyscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
        equalized = clahe.apply(grey)

        # Perform adaptive threshold
        thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

        # Find contours
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Initialize a mask that will contain bad blobs (the ones that
        # would need to be filtered by area)
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255
        # Define the minimal area of the blobs
        # MAY BE RE-THINK MINAREA
        minArea = 10.0
        width, height = frame.shape[:2]
        roiArea = width * height
        # Loop over the contours
        for c in contours:
            # Identify area of the contour
            area = cv2.contourArea(c)
            # Identify the ratio of area of blob to area of contour
            ratio = area / roiArea
            # If ratio is smaller than 30 percent
            if ratio < 0.3:
                # In case the area of the contour is smaller than
                # the minimum area, draw it on the mask
                if cv2.contourArea(c) > minArea:
                    cv2.drawContours(mask, [c], -1, 0, -1)

        # Subtract small blobs from the frame
        filtered = cv2.subtract(thresh, mask)
        # Return the mask
        return filtered

    # Determines if the blob contains glare
    def is_glare(self, blob):
        # Convert RGB frame to HSV for better colour separation
        hsv = cv2.cvtColor(blob, cv2.COLOR_BGR2HSV)

        # Define the range of white color in HSV
        # These threshold are less sensitive to glare, but produce more results
        lower_white = np.array([4, 8, 237])
        upper_white = np.array([150, 102, 255])

        # (DID NOT TURN OUT TO BE GOOD)
        # These threshold eliminate some of actual results,
        # but are good at detecting most glare
        # lower_white = np.array([4, 8, 165])
        # upper_white = np.array([150, 126, 255])

        # Threshold the HSV image to mark the glare parts as foreground
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Check if some glare was identified
        if cv2.countNonZero(mask) > 0:
            return True
        else:
            return False

    # Produces the skeleton of the blob and counts all non-zero pixels
    def line_length(self, mask):
        # Copy mask
        img = mask.copy()
        # Define size of skeleton blob
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)
        # Define structuring element for morphology
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        done = False
        # Perform skeletonisation
        while (not done):
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        # Return the number of pixels in the skeleton
        return cv2.countNonZero(skel)

    # Performs multiple mode estimation
    def get_mode(self, list):
        counter = Counter(list)
        _, val = counter.most_common(1)
        return [x for x, y in counter.items() if y == val]

    # Prints the feature table to the console and runs
    # statistical analysis on that feature table.
    def print_state(self):
        self.print_feature_table()
        self.analyse_feature_table()

    # Prints the current state of the feature table to the console
    def print_feature_table(self):
        # Create and print title
        title = '(X,Y) Coordinate |  Width  |  Height  | Orientation |  Area  |    Colour (R,G,B)   | Length |'
        print('\n' + '---------------------------------------------------------------------------------------------')
        print(title)
        print('---------------------------------------------------------------------------------------------')
        # For each row of the table
        for row in self.featureTable:
            # Get all the columns and format them
            xCoor = '{:.5}'.format(row[0][0])
            yCoor = '{:.5}'.format(row[0][1])
            xyCoor = '{:^16}'.format(' (' + xCoor + ',' + yCoor + ')')
            width = '{:^7.4}'.format(row[1])
            height = '{:^8.4}'.format(row[2])
            orientation = '{:^11.4}'.format(row[3])
            area = '{:^6.4}'.format(row[4])
            red = '{:.5}'.format(str(row[5][0]))
            green = '{:.5}'.format(str(row[5][1]))
            blue = '{:.5}'.format(str(row[5][2]))
            colour = '{:^19}'.format('(' + red + ',' + green + ',' + blue + ')')
            length = '{:^8.4}'.format(row[6])
            # Print row
            print(xyCoor + ' | ' + width + ' | ' + height + ' | ' + orientation + ' | ' + area + ' | ' + colour + ' | ' + length + ' |')
        print('---------------------------------------------------------------------------------------------')
        print('{:^76}'.format('TOTAL: ' + str(len(self.featureTable))))
        print('--------------------------------------------------------------------------------------------- \n')

    # Getter for the analysed statistical data
    def get_statistics(self):
        return self.statistics