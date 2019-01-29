from src.segmentation import *


def threshold_ui_single(dir, filename, windowname='Connected components result'):
    """
    The GUI for a single image (brain/heart) threshold adjustment
    """
    path = dir + filename

    cv2.namedWindow(windowname)

    def trackbar_callback(value):
        return value

    cv2.createTrackbar('Lower Threshold', windowname, 255, 255, trackbar_callback)
    cv2.createTrackbar('Upper Threshold', windowname, 255, 255, trackbar_callback)

    while True:
        img = cv2.imread(path, 0)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        lower_thresh = cv2.getTrackbarPos('Lower Threshold', windowname)
        upper_thresh = cv2.getTrackbarPos('Upper Threshold', windowname)

        ori, colored = connected_component(img, connectivity=10, lower_thresh=lower_thresh, upper_thresh=upper_thresh)
        ori = draw_contours(ori, colored)

        cv2.imwrite('seg_contoured_' + filename, ori)
        cv2.imwrite('seg_colored_' + filename, colored)

        cv2.imshow(windowname, np.hstack((ori, colored)))
        key = cv2.waitKey(3000)
        if key == ord('n'):
            break


def threshold_ui(dir, percentile1, percentile2, windowname='Connected components result'):
    """
    The GUI for slice-by-slice image threshold adjustment
    :param dir: the directory where the image that need to be displayed are stored
    :param percentile1: percentile for lower threshold
    :param percentile2: percentile for upper threshold
    :param windowname: display window name
    """
    os.chdir(path=dir)
    for filename in os.listdir(dir):
        if not filename.startswith('seg') and filename.endswith('.png'):
            img_path = dir + filename
            print(img_path)
            img = cv2.imread(img_path, 0)
            y_smooth = hist_gen(img)
            lower_thresh = thresh_gen_percentile(y_smooth, percentile1)

            upper_thresh = thresh_gen_percentile(y_smooth, percentile2)

            img, colored = connected_component(img,
                                               lower_thresh=lower_thresh,
                                               upper_thresh=upper_thresh)
            img = draw_contours(img, colored)

            cv2.imwrite('seg_contoured_' + filename, img)
            cv2.imwrite('seg_colored_' + filename, colored)

            cv2.imshow(windowname, np.hstack((img, colored)))
            key = cv2.waitKey(4000)
            if key == ord('p'):
                threshold_ui_single(dir=dir, filename=filename)
    cv2.destroyAllWindows()


def window_display(img1, img2, img3, img4, win):
    """
    helper function for threshold determination testing gui display
    :param img1: segmented image with threshold from histogram peaks percentile(15%-25%)
    :param img2: segmented image with threshold from histogram peaks percentile(25%-50%)
    :param img3: segmented image with threshold from histogram peaks percentile(50%-70%)
    :param img4: segmented image with threshold from histogram peaks percentile(70%-100%)
    :param win:  window name for display
    """
    img1 = cv2.resize(img1, (400, 400))
    img2 = cv2.resize(img2, (400, 400))
    img3 = cv2.resize(img3, (400, 400))
    img4 = cv2.resize(img4, (400, 400))

    _, width1, canvas1 = draw_canvas(img1, 10, 10)
    _, width2, canvas2 = draw_canvas(img2, 10, 10)
    _, width3, canvas3 = draw_canvas(img3, 10, 10)
    _, width4, canvas4 = draw_canvas(img4, 10, 10)

    text1 = "percentile(15%-25%)"
    text2 = "percentile(25%-50%)"
    text3 = "percentile(50%-70%)"
    text4 = "percentile(70%-100%)"

    img1 = cv2.putText(canvas1.copy(), text1, (int(0.25 * width1), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       (0, 100, 255))
    img2 = cv2.putText(canvas2.copy(), text2, (int(0.25 * width2), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       (0, 100, 255))
    img3 = cv2.putText(canvas3.copy(), text3, (int(0.25 * width3), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       (0, 100, 255))
    img4 = cv2.putText(canvas4.copy(), text4, (int(0.25 * width4), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       (0, 100, 255))

    cv2.namedWindow(win)
    aligned_img = np.vstack((np.hstack((img1, img2)), np.hstack((img3, img4))))

    cv2.imshow(win, aligned_img)


def thresh_tests_display(img):
    """
    threshold determination testing gui display of a single image
    :return: segmented image with threshold from histogram peaks percentile(15%-25%), (25%-50%), (50%-70%), (70%-100%)
    """
    img = cv2.GaussianBlur(img, (5, 5), 0)
    y_smooth = hist_gen(img, 40)

    p_10 = thresh_gen_percentile(y_smooth, 15)
    p_25 = thresh_gen_percentile(y_smooth, 25)
    p_50 = thresh_gen_percentile(y_smooth, 50)
    p_70 = thresh_gen_percentile(y_smooth, 70)
    p_100 = thresh_gen_percentile(y_smooth, 100)

    _, out1 = connected_component(img, 10, p_10, p_25)
    _, out2 = connected_component(img, 10, p_25, p_50)
    _, out3 = connected_component(img, 10, p_50, p_70)
    _, out4 = connected_component(img, 10, p_70, p_100)

    return out1, out2, out3, out4


def thresh_test_slice_by_slice(dir, dataset_name):
    """
    threshold determination testing gui display slice by slice
    :param dir: the directory where the image that need to be displayed are stored
    """
    for filename in os.listdir(dir):
        if filename.endswith('.png'):
            img_path = dir + filename
            img = cv2.imread(img_path, 0)
            out1, out2, out3, out4 = thresh_tests_display(img)

            win = 'Threshold tests based on percentiles: {}'.format(dataset_name)
            window_display(out1, out2, out3, out4, win)
            cv2.waitKey(3000)
    cv2.destroyAllWindows()


def seg_neuron_ui_single(dir, filename, windowname='Segmentation of neurons tests'):
    """
    The GUI for a single neuron image threshold adjustment
    :param dir: the directory where the image that need to be displayed are stored
    :param filename: each filename in the directory
    :param windowname: window name for display
    """
    path = dir + filename

    cv2.namedWindow(windowname)

    def trackbar_callback(value):
        return value

    cv2.createTrackbar('Lower Threshold', windowname, 155, 255, trackbar_callback)
    cv2.createTrackbar('Upper Threshold', windowname, 255, 255, trackbar_callback)

    while True:

        lower_thresh = cv2.getTrackbarPos('Lower Threshold', windowname)
        upper_thresh = cv2.getTrackbarPos('Upper Threshold', windowname)

        img, binary = seg_neurons(path, windowname, mode='inrange', thresh=lower_thresh, maxval=upper_thresh,
                                  channel='b')
        img = draw_contours(img, binary, channel='b')

        cv2.imwrite('seg_contoured_' + filename, img)
        cv2.imwrite('seg_colored_' + filename, binary)

        cv2.imshow(windowname, np.hstack((img, binary)))
        key = cv2.waitKey(3000)
        if key == ord('n'):
            break


def seg_neuron_test_slice_by_slice(dir, dataset_name):
    """
    The GUI for neuron images threshold adjustment slice by slice
    :param dir: the directory where the image that need to be displayed are stored
    """
    for filename in os.listdir(dir):
        if not filename.startswith('seg') and filename.endswith('.tif'):
            img_path = dir + filename

            win = 'Segmentation of neurons tests: {}'.format(dataset_name)
            img, binary = seg_neurons(img_path, win, channel='b')
            img = draw_contours(img, binary, channel='b')

            cv2.imwrite('seg_contoured_' + filename, img)
            cv2.imwrite('seg_colored_' + filename, binary)

            cv2.imshow(win, np.hstack((img, binary)))
            key = cv2.waitKey(3000)
            if key == ord('p'):
                seg_neuron_ui_single(dir=dir, filename=filename, windowname=win)
    cv2.destroyAllWindows()
