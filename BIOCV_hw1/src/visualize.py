from src.segmentation import *


def plot_images_general_case(images, num_of_images, figsize=(10, 10), titles=None):
    """
    :param images: image stack represented as multidimensional matrix
    :param num_of_images: number of images required to be displayed
    :param figsize: assigned figure size to display
    :param titles: assigned title for each image
    """
    plt.figure(figsize=figsize)
    for i in range(num_of_images):
        row = int(np.sqrt(num_of_images))
        col = int(num_of_images / row)
        plt.subplot(row, col, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def plot_RGB_samples(I_r, I_b, I_g):
    """
    :param I_r: image stack of the red channel
    :param I_b: image stack of the blue channel
    :param I_g: image stack of the gray channel
    """

    for i in range(len(I_r)):
        random_idx = np.random.choice(len(I_r), 1)[0]
        img1 = I_r[random_idx]
        img2 = I_b[random_idx]
        img3 = I_g[random_idx]

        img1 = cv2.resize(img1, (400, 400))
        img2 = cv2.resize(img2, (400, 400))
        img3 = cv2.resize(img3, (400, 400))

        _, width1, canvas1 = draw_canvas(img1, 20, 10)
        _, width2, canvas2 = draw_canvas(img2, 20, 10)
        _, width3, canvas3 = draw_canvas(img3, 20, 10)

        text1 = "I_r"
        text2 = "I_b"
        text3 = "I_g"

        img1 = cv2.putText(canvas1.copy(), text1, (int(0.45 * width1), 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           (0, 100, 255))
        img2 = cv2.putText(canvas2.copy(), text2, (int(0.45 * width2), 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           (0, 100, 255))
        img3 = cv2.putText(canvas3.copy(), text3, (int(0.45 * width3), 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           (0, 100, 255))

        cv2.imshow('Neuron Image Stack RGB samples', np.hstack((np.hstack((img1, img2)),
                                                                img3)))
        cv2.waitKey(400)
    cv2.destroyAllWindows()


def plot_dicom_samples(dcm_dims, dcm_pixel_spacing, dcm_arr, num_of_images, mode, save_path, cmap=plt.gray()):
    """
    :param dcm_dims: dimension for each dcm format image;
    :param dcm_pixel_spacing: pixel_spacing for each dcm format image;
    :param dcm_arr: the whole dcm image stack as multi-dimensional matrix
    :param num_of_images: number of images required to be displayed
    :param mode: 'save' or 'display'
    :param save_path: the path to save the png format images converted from dcm
    :param cmap: assigned cmap style for matplotlib
    """
    x = np.arange(0.0, (dcm_dims[0] + 1) * dcm_pixel_spacing[0], dcm_pixel_spacing[0])
    y = np.arange(0.0, (dcm_dims[1] + 1) * dcm_pixel_spacing[1], dcm_pixel_spacing[1])

    # plt.figure(figsize=(10, 10))

    plt.set_cmap(cmap=cmap)

    num = dcm_dims[2] if mode == 'save' else num_of_images
    for i in range(num):
        if mode != 'save':
            row = int(np.sqrt(num))
            col = np.ceil(num/row)
            plt.subplot(row, col, i + 1)
        plt.pcolormesh(x, y, np.flipud(dcm_arr[:, :, i]))
        plt.xticks([])
        plt.yticks([])
        if mode == 'save':
            file_existed = np.any(
                [True if '{}.png'.format(i) in filename else False for filename in os.listdir(save_path)])
            if not file_existed:
                plt.savefig(save_path + '{}.png'.format(i), bbox_inches='tight', pad_inches=0)
            else:
                print('{}.png already exists under the directory {}'.format(i, save_path))
    if mode != 'save':
        plt.show()

