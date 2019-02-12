import numpy as np
import cv2

PATH = '/home/karenyyy/workspace/Datasci_Spr_19/IST597BIOCV/images/'


def norm(x):
    max_val = np.max(x, axis=0)
    x = x / max_val
    return x


def euclidean_dist(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))


def transform_img_3_dim(image):
    H = image.shape[0]
    W = image.shape[1]

    data_points = np.zeros((W * H, 3))

    for h in range(0, H):
        for w in range(0, W):
            rgb = image[h, w]
            data_points[h * W + w, 0] = rgb[0]
            data_points[h * W + w, 1] = rgb[1]
            data_points[h * W + w, 2] = rgb[2]

    data_points_scaled = norm(data_points)

    return H, W, data_points_scaled


def transform_img_5_dim(image):
    H = image.shape[0]
    W = image.shape[1]

    data_points = np.zeros((W * H, 5))

    for h in range(0, H):
        for w in range(0, W):
            rgb = image[h, w]
            data_points[h * W + w, 0] = rgb[0]
            data_points[h * W + w, 1] = rgb[1]
            data_points[h * W + w, 2] = rgb[2]
            data_points[h * W + w, 3] = w
            data_points[h * W + w, 4] = h

    data_points_scaled = norm(data_points)
    return H, W, data_points_scaled


def color_clusters(image, labels, data_points, centroids):
    H = image.shape[0]
    W = image.shape[1]

    for idx, label in enumerate(labels):
        data_points[idx][0] = np.array(centroids[label])[0] * 255
        data_points[idx][1] = np.array(centroids[label])[1] * 255
        data_points[idx][2] = np.array(centroids[label])[2] * 255

    image = np.zeros((H, W, 3), np.uint8)

    for h in range(H):
        for w in range(W):
            image[h, w] = [int(data_points[h * W + w][0]),
                           int(data_points[h * W + w][1]),
                           int(data_points[h * W + w][2])]
    cv2.imwrite('segmented.png', image)
    return image


def draw_contours(ori_img, seg_img):
    """
    :param ori_img: original image
    :param seg_img: segmented image
    :param channel: if the segmented image is 3-channel image, then transform it to be grayscale image; otherwise do not do the transformation
    :return: original image with contours drawn on it
    """
    img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)

    # _, img = cv2.threshold(img, 100, 155, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 11, 10)

    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda i: len(i), reverse=True)
    contours = contours[:int(len(contours) / 4)]
    cv2.drawContours(ori_img, contours, -1, (0, 0, 255), 3)
    return ori_img
