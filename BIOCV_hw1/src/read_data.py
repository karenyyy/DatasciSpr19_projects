import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import pydicom as pd


def process_image_stack(path):
    """
    :return: Image split from R, G B channels respectively
    """
    try:
        I_r = []
        I_b = []
        I_g = []
        neuron_tiff_images_lst = os.listdir(path=path)
        for img_file in neuron_tiff_images_lst:
            if img_file.endswith('tif'):
                abs_path = path + img_file
                img_arr = cv2.imread(filename=abs_path)
                img_arr_I_b, img_arr_I_g, img_arr_I_r = cv2.split(img_arr)
                I_r.append(img_arr_I_r)
                I_b.append(img_arr_I_b)
                I_g.append(img_arr_I_g)
        return I_r, I_g, I_b
    except Exception as e:
        print(e)


def process_dicom_stack(path):
    """
    :param path: dicom_stack_path
    :return: dcm_dims: dimension for each dcm format image;
             dcm_pixel_spacing: pixel_spacing for each dcm format image;
             dcm_arr: store the whole dcm stack as multi-dimensional matrix
     note*: the parameters saved for image stack display tests
    """
    try:
        lst_dcm = []
        for filename in os.listdir(path):
            if not filename.endswith('.png'):
                lst_dcm.append(path + filename)
        dcm_config = pd.read_file(lst_dcm[0])
        dcm_dims = (int(dcm_config.Rows),
                    int(dcm_config.Columns),
                    len(lst_dcm))

        print('The slice thickness is: {}.'.format(dcm_config.SliceThickness))
        print('The pixel spacing is: {}'.format(dcm_config.PixelSpacing))

        dcm_pixel_spacing = (float(dcm_config.PixelSpacing[0]),
                             float(dcm_config.PixelSpacing[1]),
                             float(dcm_config.SliceThickness))

        dcm_arr = np.zeros(dcm_dims, dtype=dcm_config.pixel_array.dtype)

        for file in lst_dcm:
            ds = pd.read_file(file)
            dcm_arr[:, :, lst_dcm.index(file)] = ds.pixel_array

        return dcm_dims, dcm_pixel_spacing, dcm_arr
    except Exception as e:
        print(e)