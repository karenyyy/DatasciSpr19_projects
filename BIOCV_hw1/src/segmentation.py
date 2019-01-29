import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from util.utils import *


def seg_neurons(path,
                win,
                blur='gaussian',
                channel='b',
                mode='binary',
                thresh=None,
                maxval=None):
    """
    Binary image analysis for neuron image segmentation
    :param path: the directory where the neuron images are stored
    :param win: the window name displayed
    :param blur: method of image blurring, 'gaussian' or 'median'
    :param channel: channel='b', then just extract the b channel of the original image; otherwise, transform the original image to grayscale
    :param mode: the mode of thresholding. if mode == 'binary', then use OTSU; if mode == 'inrange', then use inRange
    :param thresh: parameter of OTSU thresholding
    :param maxval: parameter of OTSU thresholding
    :return: original image, binary image
    """
    im = cv2.imread(path)

    if channel == 'b':
        img, _, _ = cv2.split(im)
    else:
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if blur == 'gaussian':
        img = cv2.GaussianBlur(img, (5, 5), 0)
    elif blur == 'median':
        img = cv2.medianBlur(img, 5)

    cv2.namedWindow(win)

    if thresh is None:
        thresh = -1

    if maxval is None:
        maxval = 255

    if mode == 'binary':
        _, binary = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == 'inrange':
        binary = cv2.inRange(img, thresh, maxval)
    else:
        binary = img

    # binary = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)
    return img, binary


def seg_brain_and_heart(img):
    """
    Use thresholding and morphological operations for segmentation
    :param lower_thresh: lower threshold
    :return: different morph approaches for segmentation
     (first it started out as for testing purpose, later when connected component analysis reaches a better outcome,
        the results here were not saved, just for testing)
    """
    _, mask = cv2.threshold(img, -1, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    eroded = cv2.morphologyEx(mask, cv2.MORPH_ERODE, (3, 3), iterations=10)
    dilated = cv2.morphologyEx(mask, cv2.MORPH_DILATE, (3, 3), iterations=10)

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                              iterations=5)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                              iterations=5)

    grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    return img, mask, eroded, dilated, opened, closed, grad


def optimize_mask(img, lower_thresh, upper_thresh):
    """
    generate mask based on threshold constraints
    :param lower_thresh: lower threshold
    :param upper_thresh: upper threshold
    :return: mask
    """
    y_smooth = hist_gen(img)

    if lower_thresh == 255:
        lower_thresh = thresh_gen_simple(y_smooth)

    print('Current Lower Threshold: ', lower_thresh)
    print('Current Upper Threshold: ', upper_thresh)
    print('\n')

    mask = cv2.inRange(img, lower_thresh, upper_thresh)
    return mask


def connected_component(img, connectivity=10, lower_thresh=255, upper_thresh=255):
    """
    Applying connected component analysis for image segmentation
    :param connectivity: parameter for connected component analysis package in opencv
    :param lower_thresh: lower threshold
    :param upper_thresh: upper threshold
    :return: original image, colored segmented image
    """
    img = cv2.equalizeHist(img)
    mask = optimize_mask(img, lower_thresh, upper_thresh)

    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)

    num_labels, labelmap, stats, centers = output

    colored = np.full((img.shape[0], img.shape[1], 3), 0, np.uint8)

    for l in range(1, num_labels):
        if stats[l][4] > 200:
            colored[labelmap == l] = (0, 255 * l / num_labels, 255 * num_labels / l)
            cv2.circle(colored,
                       (int(centers[l][0]), int(centers[l][1])),
                       5, (0, 100, 0), cv2.FILLED)
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img, colored

