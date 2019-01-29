from scipy import signal
import numpy as np
import cv2

def hist_gen(img, scale=40):
    """
    generate histogram of image for threshold determination
    :param img: input image, normally in grayscale
    :param scale: smoothing scale of box filter
    :return: histogram after smoothing with box filter
    """
    hist = np.histogram(img.ravel(), 256, [0, 256])

    box = np.random.uniform(0, 1 / scale, size=scale)
    y_smooth = np.array(np.convolve(hist[0], box, mode='same'))
    return y_smooth


def thresh_gen_simple(y_smooth):
    """
    simple threshold adjustment when the auto-segmentation way based on percentile outputs suboptimal results
    :param y_smooth: histogram after smoothing with box filter
    :return: lower threshold
    """
    if len(signal.argrelextrema(y_smooth, np.greater)[0]) > 3:
        lower_thresh = np.percentile(signal.argrelextrema(y_smooth, np.greater), 70)
        if lower_thresh > 190:
            lower_thresh -= 50
        if lower_thresh < 110:
            lower_thresh += 50
    else:
        lower_thresh = 50
    return lower_thresh


def thresh_gen_percentile(y, percentile):
    """
    determine threshold based on percentile of the image histogram peaks
    :param y: historam (better input the smoothed one)
    :param percentile: (preferred percentile of the histogram peaks)
    :return: percentile of image histogram peaks
    """
    thresh_lst = list(signal.argrelextrema(y, np.greater)[0])
    return np.percentile(thresh_lst, percentile)


def draw_contours(ori_img, seg_img, channel='bgr'):
    """
    :param ori_img: original image
    :param seg_img: segmented image
    :param channel: if the segmented image is 3-channel image, then transform it to be grayscale image; otherwise do not do the transformation
    :return: original image with contours drawn on it
    """
    if channel == 'bgr':
        img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    else:
        img = seg_img

    # _, img = cv2.threshold(img, 100, 155, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 11, 10)

    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda i: len(i), reverse=True)
    contours = contours[:int(len(contours)/4)]
    cv2.drawContours(ori_img, contours, -1, (0, 0, 255), 3)
    return ori_img


def draw_canvas(image, padding_top, padding_left):
    """
    helper function for threshold determination testing gui (used in gui)
    :param image: 3-channel image
    :return: new padded h, w and padded image
    """
    try:
        height, width, channel = image.shape
        new_width, new_height = width + int(width / 20), int(height + height / 8)
        canvas = np.ones((new_height, new_width, channel), dtype=np.uint8) * 125
    except Exception as e:
        height, width = image.shape
        new_width, new_height = width + int(width / 20), int(height + height / 8)
        canvas = np.ones((new_height, new_width), dtype=np.uint8) * 125

    if padding_top + height < new_height and padding_left + width < new_width:
        canvas[padding_top:padding_top + height, padding_left:padding_left + width] = image
    else:
        print("The Given padding exceeds the limits.")

    return height, width, canvas
