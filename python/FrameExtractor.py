# Filename: FrameExtractor.py
# Author: Dmytro Poliyivets
# Institution: King's College London
# Copyright: 2018, Dmytro Poliyivets, King's College London


# Import statements
import numpy as np
import cv2


# This class performs the extraction of the frames of interest that
# match some particular characteristics from the endoscopic footage.
class FrameExtractor:

    # Class constructor
    def __init__(self, footagePath, outputFolder):
        # Define the local class variables
        # Load the endoscopic footage data (video)
        self.footage = cv2.VideoCapture(footagePath)
        # Define an output folder
        self.outputFolder = outputFolder
        # Load a 'Near Focus and NBI' template in greyscale (used for detection of NBI and focus)
        self.nearFocusNBI = cv2.imread('./resources/templatePatches/nearFocusNBI.png', 0)
        # Variables required for saving frames to disk:
        # Frame that was previously saved to disk
        self.previousFrame = None
        # Title of the previously saved frame
        self.previousFrameNumber = None
        # Counter for how many frames were saved to disk
        self.savedFrameCounter = 0

    # Identifies the frames that are not moving (i.e. still) and calls
    # a function that saves those frames to disk.

    def save_still_frames(self):
        # A checker for grabbing the frame
        grabFrame = False
        # A variable to store the still frame
        stillFrame = None

        # Get width and height of the frames
        width = int(self.footage.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.footage.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the total number of frames and frame counter
        totalFrames = int(self.footage.get(cv2.CAP_PROP_FRAME_COUNT))
        frameCount = 0

        # Print a note to the console
        print('\nVideo analysis starts...')
        print('There is a total of %d frames in the video.\n' % totalFrames)

        # Loop the whole video space
        while self.footage.isOpened():
            # Show progress to the console
            self.show_progressbar(frameCount, totalFrames)

            # Read the current frame and increment frame counter
            ret1, currentFrame = self.footage.read()
            frameCount += 1

            # Check if there are no more frames left for examination
            if currentFrame is None:
                # Close video capture object and all corresponding windows
                self.footage.release()
                cv2.destroyAllWindows()

            # Check if the frame is the near focus and that NBI is enabled
            elif self.check_nbi_and_near_focus_enabled(currentFrame, width, height):
                # Get the next frame and increment frame counter
                ret2, nextFrame = self.footage.read()
                frameCount += 1

                # Crop the frames to reduce the search space
                croppedCFrame = currentFrame[int((height * 0.4)):int((height * 0.6)), int((width * 0.45)):int((width * 0.6))]
                croppedNFrame = nextFrame[int((height * 0.4)):int((height * 0.6)), int((width * 0.45)):int((width * 0.6))]

                # Perform template matching using normalized cross-correlation
                result = cv2.matchTemplate(croppedCFrame, croppedNFrame, cv2.TM_CCOEFF_NORMED)
                # Get values of minimal and maximal match together with their locations
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # Check if two images are the same
                if max_val == 1.0:
                    # Grab the last frame
                    grabFrame = True
                    stillFrame = nextFrame
                else:
                    # Check if need to grab a frame
                    if grabFrame:
                        # Mark grab frame as false
                        grabFrame = False
                        # Crop the image to extract only the necessary part (i.e. the octagon shaped
                        # image of the esophagus)
                        croppedFrame = self.crop_frame(stillFrame)
                        # Go to the next stage of frames processing
                        self.save_to_disk(croppedFrame, frameCount, width, height)

        # Print to the console how many frames were saved
        print('\nFrames extraction finished. There is a total of %d frames saved.\n' % self.savedFrameCounter)

    # Crops the octagon shaped region that contains the video
    # footage from the rest of the elements of the frame
    def crop_frame(self, frame):
        # Define the required range of black colors in HSV space
        lower_black = np.array([0, 0, 16])
        upper_black = np.array([165, 120, 23])
        # Threshold the HSV image to mark all black regions as foreground
        mask = self.hsv_colour_threshold(frame, lower_value=lower_black, upper_value=upper_black)

        # Perform morphological closing to eliminate small objects like text and icons
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((100, 100), np.uint8))

        # Find the biggest blob...
        # Inverse the mask
        maskInverse = 255 - mask
        # Identify the contours
        _, contours, _ = cv2.findContours(maskInverse, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

        # Identify the bounding box around the largest blob
        x, y, w, h = cv2.boundingRect(contours[maxAreaIndex])

        # Crop input frame according to the side of the bounding box
        croppedFrame = frame[y:y + h, x:x + w]

        # Return the cropped frame
        return croppedFrame

    # Identifies if the frame is near focus and if NBI is enabled by
    # using template matching with an image patch which contains the
    # 'Near Focus and NBI' icon.
    def check_nbi_and_near_focus_enabled(self, frame, width, height):
        # Cut the frame to reduce the search space. Reduce to a patch that
        # contains only 10 percent of frame height and 5 percent of frame width
        croppedFrame = frame[0:int((height * 0.1)), int((width * 0.95)):width]

        # Convert the frame to greyscale
        greyFrame = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2GRAY)

        # Perform template matching using normalized cross-correlation
        result = cv2.matchTemplate(greyFrame, self.nearFocusNBI, cv2.TM_CCOEFF_NORMED)
        # Get values of minimal and maximal match together with their locations
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Check if the value of maximal match is greater than the threshold
        if max_val > 0.99:
            return True
        else:
            return False

    # Saves the frame to disk, making sure there are no duplicates
    def save_to_disk(self, frame, frameCount, width, height):
        # Check that the frame passes darkness threshold. This is required
        # to make sure that no black frames are saved to disk.
        if self.get_darkness_percent(frame) < 0.3:
            # Check if any frames were saved previously
            if self.previousFrame is not None:
                # Perform duplicate check...
                # Crop the frames to decrease the search space
                croppedCurr = frame[int((height * 0.4)):int((height * 0.6)), int((width * 0.45)):int((width * 0.6))]
                croppedPrev = self.previousFrame[int((height * 0.4)):int((height * 0.6)), int((width * 0.45)):int((width * 0.6))]

                # Perform template matching using normalized cross-correlation
                # between the cropped versions current and the previously saved frame
                result = cv2.matchTemplate(croppedCurr, croppedPrev, cv2.TM_CCOEFF_NORMED)
                # Get values of minimal and maximal match together with their locations
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # Check if images are within the similarity threshold value
                if max_val > 0.6:
                    # Record export of the current frame
                    self.previousFrame = frame
                    # Over-write previous frame with this as it is the most sharp
                    cv2.imwrite(self.outputFolder+'frame%d.png' % self.previousFrameNumber, frame)
                # Images are different just save as usual
                else:
                    # Record export of the current frame
                    self.previousFrame = frame
                    # Record the number of this frame
                    self.previousFrameNumber = frameCount
                    # Over-write previous frame with this as it is the most sharp
                    cv2.imwrite(self.outputFolder+'frame%d.png' % frameCount, frame)
                    # Update saved frame counter
                    self.savedFrameCounter += 1
            # If this is the first frame to export
            else:
                # Record export of the current frame
                self.previousFrame = frame
                # Record the number of this frame
                self.previousFrameNumber = frameCount
                # Write this frame to disk
                cv2.imwrite(self.outputFolder+'frame%d.png' % frameCount, frame)
                # Update saved frame counter
                self.savedFrameCounter += 1

    # Identifies the percentage of the frame that is very dark.
    # Used to make sure no dark frames are saved to disk.
    def get_darkness_percent(self, frame):
        # Define the range of white color in HSV
        lower_dark = np.array([0, 0, 30])
        upper_dark = np.array([30, 55, 37])

        # Threshold the HSV image to mark the glare parts as foreground
        mask = self.hsv_colour_threshold(frame, lower_value=lower_dark, upper_value=upper_dark)

        # Calculate frame darkness percentage
        height, width, channels = frame.shape
        darkness = cv2.countNonZero(mask) / (width * height)

        return darkness

    # Performs HSV colour threshold on the image based on the upper and
    # lower threshold values provided and returns the resulting mask.
    def hsv_colour_threshold(self, frame, lower_value, upper_value):
        # Convert RGB frame to HSV for better colour separation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to mark all black regions as foreground
        mask = cv2.inRange(hsv, lower_value, upper_value)

        return mask

    # Prints the progress of video analysis to the console.
    def show_progressbar(self, iteration, total, fill='█'):
        # The code taken from:
        # https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
        percent = ("{0:." + str(1) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(50 * iteration // total)
        bar = fill * filledLength + '-' * (50 - filledLength)
        print('\r%s |%s| %s%% %s' % ('Progress', bar, percent, 'Complete'), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()