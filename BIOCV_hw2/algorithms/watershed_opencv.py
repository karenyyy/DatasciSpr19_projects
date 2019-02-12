import numpy as np
import cv2

'''
Try Watershed using OpenCV
borrow code from:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
'''

class WatershedOpenCV:
    def __init__(self, X):
        self.X = X

    def apply_threshold(self):
        gray = cv2.cvtColor(self.X, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def get_background(self, binary):
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)
        background = cv2.dilate(opening, kernel, iterations=3)
        return opening, background

    def get_foreground(self, opening):
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, foreground = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        foreground = np.uint8(foreground)
        return foreground

    def get_boundary(self, background, foreground):
        boundary = cv2.subtract(background, foreground)
        return boundary

    def get_markers(self, foreground, boundary):
        _, markers = cv2.connectedComponents(foreground)

        markers = markers + 1
        markers[boundary == 255] = 0

        markers = cv2.watershed(self.X, markers)
        self.X[markers == -1] = [0, 0, 255]

