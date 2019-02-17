# Filename: FeatureDetector.py
# Author: Dmytro Poliyivets
# Institution: King's College London
# Copyright: 2018, Dmytro Poliyivets, King's College London


# Import statements
from os.path import isfile, join
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
import cv2


# The class is intended for detection of IPCL structures in a scene and
# extraction of features from those IPCLs. The extracted features include:
# (X, Y) Coordinate in the scene, Width, Height, Rotation, Area, Colour (R,G,B),
# Length, and per frame Occurrence (Total Number) of IPCLs.
class FeatureDetector:

    # Class constructor
    ########## 12/02/2019 added output_folder
    def __init__(self, input_folder, output_folder):
        # Define an input folder which contains the frames that has to be analysed
        self.input_folder = input_folder
        ########## Define an output folder to save detected features
        self.output_folder = output_folder
        # Define a table that will store extracted features, with structure:
        # [['(X, Y) Coordinate', 'Width', 'Height', 'Rotation', 'Area', 'Colour (R,G,B)', 'Length']]
        self.featureTable = []
        ########## 12/02/09
        self.feature_tables = list()
        # Define a list that contains number of IPCLs per each frame
        self.numberIPCLs = []

    # Main functions which starts feature detection
    def run(self):
        # Get all files from the detected frames directory
        files = [f for f in listdir(self.input_folder) if isfile(join(self.input_folder, f))]

        # Define the counter for the progress bar
        counter = 0

        # Print a note to the console
        print('\nFeature extraction starts...')
        print('\nIf you wish to see the detected IPCLs uncomment matplotlib instructions at the end of the extract_features function in FeatureDetector.py')
        print('\nIf you wish to see the printout of the feature table uncomment line 329 in the extract_features function in FeatureDetector.py\n')

        # For all files in the detected frames directory
        for file in files:
            # Check that current file has .png extension
            if file.endswith('.png') or file.endswith('.jpg'):
                # Load each image
                frame = cv2.imread(self.input_folder + str(file))

                # Print progress bar to the console and increment counter
                self.show_progressbar(counter, len(files))
                counter += 1

                # Runs feature extraction process on each frame
                # storing results in a feature table
                self.extract_features(str(file), frame, self.adaptive_threshold_clustering(frame))


    # Finds the four triangles in each corner and mask them. Returns the
    # resulting binary image.
    def mask_corners(self, frame):
        # Define the range of white color in HSV
        ########## 06/02/2019
        # lower_black = np.array([0, 0, 0])
        # upper_black = np.array([165, 105, 20])
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([0, 0, 0])

        # Threshold the HSV image to mark the glare parts as foreground
        # mask = self.hsv_colour_threshold(frame, lower_black, upper_black)
        mask = cv2.inRange(frame, lower_black, upper_black)
        ###################################################
        # Perform some morphology...
        # Perform morphological closing to get rid of holes inside the mask
        reducedHoles = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8))
        # Perform dilation to increase the area of the blobs a little bit
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilated = cv2.dilate(reducedHoles, kernel, iterations=1)

        # Return the mask
        return dilated

    # Finds glare regions in the frame and performs some morphology on those blobs
    # returns a mask containing filtered glare regions
    # def mask_glare(self, frame):
    #     # Define the range of white color in HSV
    #     lower_white = np.array([4, 8, 237])
    #     upper_white = np.array([150, 102, 255])
    #
    #     # Threshold the HSV image to mark the glare parts as foreground
    #     mask = self.hsv_colour_threshold(frame, lower_white, upper_white)
    #
    #     # Perform some morphology...
    #     # Perform dilation to increase the area of the blobs a little bit
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    #     dilated = cv2.dilate(mask, kernel, iterations=1)
    #     # Perform morphological closing to get rid of holes inside the mask
    #     reducedHoles = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    #     # Perform flood fill to close all holes that are left
    #     floodFill = reducedHoles.copy()
    #     h, w = reducedHoles.shape[:2]
    #     temp = np.zeros((h + 2, w + 2), np.uint8)
    #     cv2.floodFill(floodFill, temp, (0, 0), 255)
    #     floodFillInv = cv2.bitwise_not(floodFill)
    #     noHoles = reducedHoles | floodFillInv
    #
    #     # Return the result
    #     return noHoles

    def mask_glare(self, frame):
        # Define the range of white color in HSV
        lower_white = np.array([4, 8, 237])
        upper_white = np.array([150, 102, 255])

        # Threshold the HSV image to mark the glare parts as foreground
        mask = self.hsv_colour_threshold(frame, lower_white, upper_white)

        # Perform some morphology...
        # Perform dilation to increase the area of the blobs a little bit
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        # # Perform morphological closing to get rid of holes inside the mask
        reducedHoles = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
        # # Perform flood fill to close all holes that are left
        # floodFill = reducedHoles.copy()
        # h, w = reducedHoles.shape[:2]
        # temp = np.zeros((h + 2, w + 2), np.uint8)
        # cv2.floodFill(floodFill, temp, (0, 0), 255)
        # floodFillInv = cv2.bitwise_not(floodFill)
        # noHoles = reducedHoles | floodFillInv

        # Return the result
        return reducedHoles

    # Performs adaptive threshold on the whole frame to identify
    # clusters of blobs that are close together, i.e. finds ROI
    def adaptive_threshold_clustering(self, frame):
        # Convert the frame to greyscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Perform contrast limited adaptive histogram equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(grey)

        # Apply some Gaussian blur to eliminate noise
        grey_blur = cv2.GaussianBlur(equalized, (15, 15), 0)

        # Perform adaptive threshold
        thresh = cv2.adaptiveThreshold(grey_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Return the mask
        return thresh

    # Performs adaptive threshold on the ROI to identify vein structures
    def adaptive_threshold_roi(self, frame):
        # Convert the frame to greyscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        equalized = clahe.apply(grey)

        # Perform adaptive threshold
        thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 13)

        # Identify the contours in the mask
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Initialize a mask that will contain bad blobs (the ones that
        # would need to be filtered by area)
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255
        # Define the minimal area of the blobs
        # MAY BE RE-THINK MINAREA
        minArea = 10.0
        # Loop over the contours
        for c in contours:
            # In case the area of the contour is smaller than
            # the minimum area, draw it on the mask
            if cv2.contourArea(c) > minArea:
                cv2.drawContours(mask, [c], -1, 0, -1)

        filtered = cv2.subtract(thresh, mask)

        # To illustrate the results uncomment the lines below
        # fig = plt.figure()
        # plt.subplot(221)
        # plt.imshow(grey, cmap='gray')
        # plt.title('Passed Frame (Gray)')
        # plt.subplot(222)
        # plt.imshow(equalized, cmap='gray')
        # plt.title('CLAHE')
        # plt.subplot(223)
        # plt.imshow(thresh)
        # plt.title('Adaptive Thresholding')
        # plt.subplot(224)
        # plt.imshow(filtered)
        # plt.title('Filter by Min Area')
        # plt.suptitle('Step 3. Perform Blob Extraction.', fontsize=16)
        # plt.show(block=False)

        # Return the mask
        return filtered

    # Finds the region of interest that has the most concentration of blobs
    # and returns it together with the percentage value of the roi relative
    # to the initial frame size
    def get_roi(self, frame, mask):
        # Identify the four rectangles in each corner
        cornersBinary = self.mask_corners(frame)
        # Identify glare in the frame
        glareBinary = self.mask_glare(frame)
        # Combine the two masks
        subtractMask = cv2.add(cornersBinary, glareBinary)

        # Subtract the masks defined above from the frame
        filtered = cv2.subtract(mask, subtractMask)

        # Perform some morphology...
        # Perform morphological closing to group together blobs that are close to each other
        clustered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
        # Perform morphological opening to eliminate small outliers
        opening = cv2.morphologyEx(clustered, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))


        # Find the biggest blob...
        # Get the contours
        _, contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Initialize necessary variables
        maxArea, maxAreaIndex = 0, 0
        # Loop over the contours
        for index, c in enumerate(contours):
            # Get area of the current blob
            area = cv2.contourArea(c)
            # Check if the area of current blob is greater than previous maxArea
            if area > maxArea:
                # Record this area as maxArea and record its index
                maxArea = area
                maxAreaIndex = index

        # Create an empty mask of the size of original frame
        temp = np.zeros_like(clustered)

        # Draw the blobs on the frame and the mask filling the holes
        cv2.drawContours(temp, contours, maxAreaIndex, 255, -1)

        # Overlay the inverse of mask over the frame to eliminate
        # all parts that are not necessary for analysis
        maskInverse = 255 - temp
        img = frame.copy()
        _, contours, _ = cv2.findContours(maskInverse, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, 0, -1)

        # Crop the frame and the mask so that it has only the ROI
        (x, y) = np.where(temp == 255)
        if len(x) == 0 or len(y) == 0:
            (topX, topY) = (1, 1)
            (bottomX, bottomY) = (1, 1)
        else:
            (topX, topY) = (np.min(x), np.min(y))
            (bottomX, bottomY) = (np.max(x), np.max(y))
        img = img[topX:bottomX + 1, topY:bottomY + 1]
        maskInverse = maskInverse[topX:bottomX + 1, topY:bottomY + 1]

        # To illustrate the results uncomment the lines below
        fig = plt.figure()
        plt.subplot(231)
        plt.imshow(frame)
        plt.title('Original Frame')
        plt.subplot(232)
        plt.imshow(cornersBinary)
        plt.title('Mask Triangles In Each Corner')
        plt.subplot(233)
        plt.imshow(glareBinary)
        plt.title('Mask Glare')
        plt.subplot(234)
        plt.imshow(subtractMask)
        plt.title('Subtract Mask')
        plt.subplot(235)
        plt.imshow(opening)
        plt.title('Subtract Constraints & Perform Morphology')
        plt.subplot(236)
        plt.imshow(img)
        plt.title('ROI')
        plt.suptitle('Step 2. Extraction of ROI.', fontsize=16)
        # If multiple matplotlib windows are opened at the same
        # time uncomment the block=False line
        plt.show()#(block=False)

        # Return the ROI together with the mask that corresponds to that roi
        # and the percentage value of the roi relative to the initial frame size
        return img, maskInverse, img.size / frame.size

    # Counts the number of blobs in the frame, and for each
    # blob identifies it's length and colour
    ########## 12/02/2019 Added name of file to save the detected features images
    def extract_features(self, name, frame, mask):
        # Get the current region of interest
        roi, mask, percentageROI = self.get_roi(frame, mask)

        # Only perform feature extraction if the extracted ROI
        # contains at least 5 percent of the frame
        if percentageROI > 0.05:

            # Threshold the necessary data from the roi
            binaryROI = self.adaptive_threshold_roi(roi)

            # Subtract the inverse of the roi mask from the threshold
            # image to eliminate border edges
            binaryROI = cv2.subtract(binaryROI, mask)

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
                    # row = [rect[0], rect[1][0], rect[1][1], rect[2], area, meanColour, length]
                    row = [rect[0], rect[1][0], rect[1][1], rect[2], area, length]
                    # Add that new row to the feature table
                    self.featureTable.append(row)

                    # Get the bounding box and draw it on the roi copy
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(output, [box], 0, (0, 0, 255), 2)

            ########## 12/02/09
            # Add tables to the set and then empty the set to separate the data between frames
            # This way we can separate the data for each image and analyse each image data separately
            self.feature_tables.append(self.featureTable)
            self.featureTable = []

            # Record the number of IPCLs per current frame in the state (numberIPCLs)
            self.numberIPCLs.append(len(contours))

            # Print current state of the feature table to console (uncomment if need)
            # self.print_feature_table()

            # Save images for inspection. Can be used to assess accuracy.
            cv2.imwrite(self.output_folder + '_roi_' + name, roi)
            cv2.imwrite(self.output_folder + '_binary_roi_' + name, binaryROI)
            cv2.imwrite(self.output_folder + '_final_output_' + name, output)

            # To illustrate the results uncomment the lines below
            fig = plt.figure()
            plt.subplot(131)
            plt.imshow(frame)
            plt.title('Original')
            plt.subplot(132)
            plt.imshow(roi)
            plt.title('ROI')
            plt.subplot(133)
            plt.imshow(output)
            plt.title('Extracted Features')
            plt.suptitle('Step 4. Extracting Features From Blobs.', fontsize=16)
            plt.show()

    # Checks if detected IPCL is actually the glare, i.e. if some
    # small glare region was not excluded by glare detection software.
    # Happens at feature extraction stage.
    def is_glare(self, blob):
        # Define the range of white color in HSV
        # These threshold are less sensitive to glare, but produce more results
        # lower_white = np.array([4, 8, 237])
        # upper_white = np.array([150, 102, 255])

        # If you want all of the glare to be deleted, uncomment the
        # ranges below and comment the ranges above.
        # However, this did not turn out to be efficient as a lot
        # of actual IPCL data is lost.
        lower_white = np.array([4, 8, 165])
        upper_white = np.array([150, 126, 255])

        # Threshold the HSV image to mark the glare parts as foreground
        mask = self.hsv_colour_threshold(blob, lower_white, upper_white)

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
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        done = False
        # Produce topological skeleton
        while not done:
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

    # Performs HSV colour threshold on the image based on the upper and
    # lower threshold values provided and returns the resulting mask.
    def hsv_colour_threshold(self, frame, lower_value, upper_value):
        # Convert RGB frame to HSV for better colour separation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_value, upper_value)

        return mask

    # Prints the current state of the feature table
    def print_feature_table(self):
        # Create and print title
        title = '(X,Y) Coordinate |  Width  |  Height  | Orientation |  Area  | Length |'
        # For each row of the table
        for table in self.feature_tables:
            print('\n' + '---------------------------------------------------------------------------------------------')
            print(title)
            print('---------------------------------------------------------------------------------------------')
            for row in table:
                # Get all the columns and format them
                xCoor = '{:.5}'.format(row[0][0])
                yCoor = '{:.5}'.format(row[0][1])
                xyCoor = '{:^16}'.format(' (' + xCoor + ',' + yCoor + ')')
                width = '{:^7.4}'.format(row[1])
                height = '{:^8.4}'.format(row[2])
                orientation = '{:^11.4}'.format(row[3])
                area = '{:^6.4}'.format(row[4])
                length = '{:^8.4}'.format(row[5])
                # Print row
                print(xyCoor + ' | ' + width + ' | ' + height + ' | ' + orientation + ' | ' + area + ' | ' + length + ' |')
            print('---------------------------------------------------------------------------------------------')
            # print('{:^76}'.format('TOTAL: ' + str(len(self.featureTable))))
            print('--------------------------------------------------------------------------------------------- \n')

    # Getter for feature table
    def get_feature_table(self):
        return self.featureTable

    ########## 14/02/2019
    def get_feature_tables(self):
        return self.feature_tables

    # Getter for list that contains number of
    # IPCLs per each examined frame
    def get_number_ipcls(self):
        return self.numberIPCLs

    # Prints the progress bar of video analysis to the console.
    def show_progressbar(self, iteration, total, fill='█'):
        # The code is taken from:
        # https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
        percent = ("{0:." + str(1) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(50 * iteration // total)
        bar = fill * filledLength + '-' * (50 - filledLength)
        print('\r%s |%s| %s%% %s' % ('Progress', bar, percent, 'Complete'), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()

    ########## 12/02/2019
    def save_feature_table(self):
        new_file = open(self.output_folder + 'feature_table.txt', 'w+')
        title = 'Area  | Length |\n'
        for table in self.feature_tables:
            new_file.write('-------------------------------------------------------------------------\n')
            new_file.write(title)
            new_file.write('-------------------------------------------------------------------------\n')
            for row in table:
                # Get all the columns and format them
                xCoor = '{:.5}'.format(row[0][0])
                yCoor = '{:.5}'.format(row[0][1])
                xyCoor = '{:^16}'.format(' (' + xCoor + ',' + yCoor + ')')
                width = '{:^7.4}'.format(row[1])
                height = '{:^8.4}'.format(row[2])
                orientation = '{:^11.4}'.format(row[3])
                area = '{:^6.4}'.format(row[4])
                length = '{:^8.4}'.format(row[5])
                new_file.write(xyCoor + ' | ' + width + ' | ' + height + ' | ' + orientation + ' | ' + area + ' | ' + length + ' |\n')
            new_file.write('-------------------------------------------------------------------------\n')
            new_file.write('\n')
        new_file.close()
        return 0



