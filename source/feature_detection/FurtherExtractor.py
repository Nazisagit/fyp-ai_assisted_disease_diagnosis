# Filename: FurtherExtractor.py
# Author: Junkai Liao
# Institution: King's College London
# Version: 04/08/2018

import cv2
from os import listdir


class FurtherExtractor:
    
    def __init__(self, CroppedImagesPath, outputFolderPath_FurtherCroppedImages):
        self.CroppedImagesPath = CroppedImagesPath
        self.outputFolderPath_FurtherCroppedImages = outputFolderPath_FurtherCroppedImages
        
    def run(self):
        print('\nStart to further crop images...')
        files = listdir(self.CroppedImagesPath)
        
        # Set the step length for sliding of the square frame.
        step_length = 100
        
        # Avoid that step length is so large that no image outputs. Create an adaptive step length.
        adaptive_step_length = step_length
        outputs_counter = 0
        
        # Execute the FurtherExtractor and return the number of outputs.
        outputs_counter = self.FurtherExtractor(files, adaptive_step_length, outputs_counter)
        
        for adapter in range(0, 8):
            if outputs_counter == 0:
                # Adjust the adaptive step length.
                adaptive_step_length = adaptive_step_length - 10
                # Execute the FurtherExtractor and return the number of outputs.
                outputs_counter = self.FurtherExtractor(files, adaptive_step_length, outputs_counter)
                
        outputs_counter = 0
        
        print('Completed.')     

    def FurtherExtractor(self, files, step_length, outputs_counter):
        for file in files:
            # Check whether the file is '.png' or'.jpg'
            # 29/03/2019
            if file.endswith('.jpg'):
                # Load the file
                img = cv2.imread(self.CroppedImagesPath + str(file))
                
                filename = file[0:len(file)-4]
                
                height, width = img.shape[:2]
                
                counter = 1
                
                for h in range(0, height-224, step_length):
                    for w in range(0, width-224, step_length):
                        temp = img[h:h+224, w:w+224]
                        if not (temp[:,:,0]==0).any():
                            frame = temp
                            cv2.imwrite(self.outputFolderPath_FurtherCroppedImages + filename + '_%d.jpg' % counter, frame)
                            counter += 1
                            outputs_counter += 1
                counter = 1    
        return outputs_counter