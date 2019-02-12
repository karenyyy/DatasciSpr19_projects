import numpy as np
import cv2

'''
Try GrabCut using OpenCV
borrow code from: 
https://docs.opencv.org/trunk/d8/d83/tutorial_py_grabcut.html
'''

class GrabcutOpenCV:
    def __init__(self, X):
        self.X = X

    def grabcut(self):
        mask = np.zeros(self.X.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (50, 50, 490, 490)
        cv2.grabCut(self.X, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        self.X = self.X * mask2[:, :, np.newaxis]
