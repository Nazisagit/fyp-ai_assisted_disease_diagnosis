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
    # 12/02/2019 added output_folder
    def __init__(self, input_folder, output_folder):
        # Define an input folder which contains the frames that has to be analysed
        self.input_folder = input_folder
        # Define an output folder to save detected features
        self.output_folder = output_folder
        # Define a table that will store extracted features, with structure:
        # [['(X, Y) Coordinate', 'Width', 'Height', 'Rotation', 'Area', 'Colour (R,G,B)', 'Length']]
        self.feature_table = list()
        # 12/02/09
        self.feature_tables = list()
        # Define a list that contains number of IPCLs per each frame
        self.number_ipcls = list()

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
        # 06/02/2019
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([165, 105, 20])

        mask = self.hsv_colour_threshold(frame, lower_black, upper_black)
        # Perform some morphology...
        # Perform morphological closing to get rid of holes inside the mask
        reduced_holes = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((50, 50), np.uint8))
        # Perform dilation to increase the area of the blobs a little bit
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        dilated = cv2.dilate(reduced_holes, kernel, iterations=1)

        # Return the mask
        return dilated

    def mask_glare(self, frame):
        # Define the range of white color in HSV
        lower_white = np.array([4, 8, 237])
        upper_white = np.array([150, 102, 255])

        # Threshold the HSV image to mark the glare parts as foreground
        mask = self.hsv_colour_threshold(frame, lower_white, upper_white)

        # Perform some morphology...
        # Perform dilation to increase the area of the blobs a little bit
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        # # Perform morphological closing to get rid of holes inside the mask
        reduced_holes = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
        # # Perform flood fill to close all holes that are left
        flood_fill = reduced_holes.copy()
        h, w = reduced_holes.shape[:2]
        temp = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood_fill, temp, (0, 0), 255)
        flood_fill_inv = cv2.bitwise_not(flood_fill)
        no_holes = reduced_holes | flood_fill_inv

        # Return the result
        return no_holes

    # Performs adaptive threshold on the whole frame to identify
    # clusters of blobs that are close together, i.e. finds ROI
    @staticmethod
    def adaptive_threshold_clustering(frame):
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
    @staticmethod
    def adaptive_threshold_roi(frame):
        # Convert the frame to greyscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        equalized = clahe.apply(grey)

        # Perform adaptive threshold
        thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 13)

        # Identify the contours in the mask
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Initialize a mask that will contain bad blobs (the ones that
        # would need to be filtered by area)
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255
        # Define the minimal area of the blobs
        # MAY BE RE-THINK MINAREA
        min_area = 10.0
        # Loop over the contours
        for c in contours:
            # In case the area of the contour is smaller than
            # the minimum area, draw it on the mask
            if cv2.contourArea(c) > min_area:
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
        corners_binary = self.mask_corners(frame)
        # Identify glare in the frame
        glare_binary = self.mask_glare(frame)
        # Combine the two masks
        subtract_mask = cv2.add(corners_binary, glare_binary)

        # Subtract the masks defined above from the frame
        filtered = cv2.subtract(mask, subtract_mask)

        # Perform some morphology...
        # Perform morphological closing to group together blobs that are close to each other
        clustered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
        # Perform morphological opening to eliminate small outliers
        opening = cv2.morphologyEx(clustered, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))

        # Find the biggest blob...
        # Get the contours
        contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Initialize necessary variables
        max_area, max_area_index = 0, 0
        # Loop over the contours
        for index, c in enumerate(contours):
            # Get area of the current blob
            area = cv2.contourArea(c)
            # Check if the area of current blob is greater than previous maxArea
            if area > max_area:
                # Record this area as maxArea and record its index
                max_area = area
                max_area_index = index

        # Create an empty mask of the size of original frame
        temp = np.zeros_like(clustered)

        # Draw the blobs on the frame and the mask filling the holes
        cv2.drawContours(temp, contours, max_area_index, 255, -1)

        # Overlay the inverse of mask over the frame to eliminate
        # all parts that are not necessary for analysis
        mask_inverse = 255 - temp
        img = frame.copy()
        contours, _ = cv2.findContours(mask_inverse, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, 0, -1)

        # Crop the frame and the mask so that it has only the ROI
        x, y = np.where(temp == 255)
        if len(x) == 0 or len(y) == 0:
            top_x, top_y = 1, 1
            bottom_x, bottom_y = 1, 1
        else:
            top_x, top_y = np.min(x), np.min(y)
            bottom_x, bottom_y = np.max(x), np.max(y)
        img = img[top_x:bottom_x + 1, top_y:bottom_y + 1]
        mask_inverse = mask_inverse[top_x:bottom_x + 1, top_y:bottom_y + 1]

        # To illustrate the results uncomment the lines below
        # fig = plt.figure()
        # plt.subplot(231)
        # plt.imshow(frame)
        # plt.title('Original Frame')
        # plt.subplot(232)
        # plt.imshow(corners_binary)
        # plt.title('Mask Triangles In Each Corner')
        # plt.subplot(233)
        # plt.imshow(glare_binary)
        # plt.title('Mask Glare')
        # plt.subplot(234)
        # plt.imshow(subtract_mask)
        # plt.title('Subtract Mask')
        # plt.subplot(235)
        # plt.imshow(opening)
        # plt.title('Subtract Constraints & Perform Morphology')
        # plt.subplot(236)
        # plt.imshow(img)
        # plt.title('ROI')
        # plt.suptitle('Step 2. Extraction of ROI.', fontsize=16)
        # If multiple matplotlib windows are opened at the same
        # time uncomment the block=False line
        # plt.show()#(block=False)

        # Return the ROI together with the mask that corresponds to that roi
        # and the percentage value of the roi relative to the initial frame size
        return img, mask_inverse, img.size / frame.size

    # Counts the number of blobs in the frame, and for each
    # blob identifies it's length and colour
    # 12/02/2019 Added name of file to save the detected features images
    def extract_features(self, name, frame, mask):
        # Get the current region of interest
        roi, mask, percentage_roi = self.get_roi(frame, mask)

        # Only perform feature extraction if the extracted ROI
        # contains at least 5 percent of the frame
        if percentage_roi > 0.05:

            # Threshold the necessary data from the roi
            binary_roi = self.adaptive_threshold_roi(roi)

            # Subtract the inverse of the roi mask from the threshold
            # image to eliminate border edges
            binary_roi = cv2.subtract(binary_roi, mask)

            # Perform feature extraction for each blob...
            # Identify the contours in the binary roi and copy the roi
            contours, _ = cv2.findContours(binary_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            output = roi.copy()
            # Loop all contours
            for c in contours:
                # Check if blob contains glare
                # Get blob dimensions and location
                x, y, w, h = cv2.boundingRect(c)
                # Crop the image containing the blob from ROI
                blob_mask = np.zeros(roi.shape[:2], np.uint8)
                cv2.drawContours(blob_mask, [c], -1, 255, -1)
                blob = cv2.bitwise_and(roi, roi, mask=blob_mask)
                blob = blob[y:y + h, x:x + w]

                # Make sure blob has no glare
                if not self.is_glare(blob):
                    # Create a mask containing this blob
                    mask = np.zeros(binary_roi.shape, np.uint8)
                    cv2.drawContours(mask, [c], 0, 255, -1)

                    # Get the mean BGR colour of the blob
                    blue, green, red, _ = cv2.mean(roi, mask=mask)
                    mean_colour = (red, green, blue)

                    # Get the bounding rectangle around the blob, taking into
                    # account its area and rotation
                    rect = cv2.minAreaRect(c)

                    # Get the area of the current blob
                    area = cv2.contourArea(c)

                    # Get the length of the contour
                    length = float(self.line_length(mask))

                    # [['Width', 'Height', 'Rotation', 'Area', 'Colour (R,G,B)', 'Length']]
                    row = [rect[1][0], rect[1][1], rect[2], area, mean_colour, length]

                    # Add that new row to the feature table
                    self.feature_table.append(row)

                    # Get the bounding box and draw it on the roi copy
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(output, [box], 0, (0, 0, 255), 2)

            # 12/02/09
            # Add tables to the set and then empty the set to separate the data between frames
            # This way we can separate the data for each image and analyse each image data separately
            self.feature_tables.append(self.feature_table)
            self.feature_table = []

            # Record the number of IPCLs per current frame in the state (numberIPCLs)
            self.number_ipcls.append(len(contours))

            # Print current state of the feature table to console (uncomment if need)
            # self.print_feature_table()

            # Save images for inspection. Can be used to assess accuracy.
            cv2.imwrite(self.output_folder + '_roi_' + name, roi)
            cv2.imwrite(self.output_folder + '_binary_roi_' + name, binary_roi)
            cv2.imwrite(self.output_folder + '_final_output_' + name, output)

            # To illustrate the results uncomment the lines below
            # fig = plt.figure()
            # plt.subplot(131)
            # plt.imshow(frame)
            # plt.title('Original')
            # plt.subplot(132)
            # plt.imshow(roi)
            # plt.title('ROI')
            # plt.subplot(133)
            # plt.imshow(output)
            # plt.title('Extracted Features')
            # plt.suptitle('Step 4. Extracting Features From Blobs.', fontsize=16)
            # plt.show()

    # Checks if detected IPCL is actually the glare, i.e. if some
    # small glare region was not excluded by glare detection software.
    # Happens at feature extraction stage.
    def is_glare(self, blob):
        # Define the range of white color in HSV
        # These threshold are less sensitive to glare, but produce more results
        lower_white = np.array([4, 8, 237])
        upper_white = np.array([150, 102, 255])

        # If you want all of the glare to be deleted, uncomment the
        # ranges below and comment the ranges above.
        # However, this did not turn out to be efficient as a lot
        # of actual IPCL data is lost.
        # lower_white = np.array([4, 8, 165])
        # upper_white = np.array([150, 126, 255])

        # Threshold the HSV image to mark the glare parts as foreground
        mask = self.hsv_colour_threshold(blob, lower_white, upper_white)

        # Check if some glare was identified
        if cv2.countNonZero(mask) > 0:
            return True
        else:
            return False

    # Produces the skeleton of the blob and counts all non-zero pixels
    @staticmethod
    def line_length(mask):
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
    @staticmethod
    def hsv_colour_threshold(frame, lower_value, upper_value):
        # Convert RGB frame to HSV for better colour separation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_value, upper_value)

        return mask

    # Prints the current state of the feature table
    def print_feature_table(self):
        # Create and print title
        title = '  Rotation  |  Area  |  Length  |  Width  |  Height  |'
        # For each row of the table
        for table in self.feature_tables:
            print('\n---------------------------------------------------------------------------------------------')
            print(title)
            print('---------------------------------------------------------------------------------------------')
            for row in table:
                width = '{:^7.4}'.format(row[1])
                height = '{:^8.4}'.format(row[2])
                rotation = '{:^11.4}'.format(row[3])
                area = '{:^6.4}'.format(row[4])
                red = '{:.5}'.format(str(row[5][0]))
                green = '{:.5}'.format(str(row[5][1]))
                blue = '{:.5}'.format(str(row[5][2]))
                colour = '{:^19}'.format('(' + red + ',' + green + ',' + blue + ')')
                length = '{:^8.4}'.format(row[6])
                print(width + ' | ' + height + ' | ' + rotation + ' | ' + area + ' | ' + colour + ' | ' + length + ' |')
            print('---------------------------------------------------------------------------------------------')
            print('{:^76}'.format('TOTAL: ' + str(len(table))))
            print('--------------------------------------------------------------------------------------------- \n')

    # 14/02/2019
    def get_feature_tables(self):
        return self.feature_tables

    # Getter for list that contains number of
    # IPCLs per each examined frame
    def get_number_ipcls(self):
        return self.number_ipcls

    # Prints the progress bar of video analysis to the console.
    @staticmethod
    def show_progressbar(iteration, total, fill='█'):
        # The code is taken from:
        # https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
        percent = ("{0:." + str(1) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(50 * iteration // total)
        bar = fill * filled_length + '-' * (50 - filled_length)
        print('\r%s |%s| %s%% %s' % ('Progress', bar, percent, 'Complete'), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()
