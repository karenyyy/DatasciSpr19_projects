from gui.gui import *
from src.segmentation import *
from src.read_data import *
from src.visualize import *

import argparse

parser = argparse.ArgumentParser(description='User input parameters')

parser.add_argument('--path',
                    default='/home/karenyyy/workspace/Datasci_Spr_19/IST597BIOCV/hw/dataset/',
                    help='path where the neuron, brain and heart datasets are stored')
parser.add_argument('--samples',
                    default=9,
                    help='number of dicom samples to display')
parser.add_argument('--testcase',
                    default=1,
                    help='which test case to test')
parser.add_argument('--mode',
                    default='save',
                    help='specifically for testcase 2, save/display')
args = parser.parse_args()

PATH = args.path
num_samples = int(args.samples)
test_case = float(args.testcase)
mode = args.mode

PATH_TIF = PATH + 'Neuron_TIff_sequence/'
PATH_BRAIN_SET1_DICOM = PATH + 'Brain_tumor/Brain_tumor/set1/'
PATH_BRAIN_SET2_DICOM = PATH + 'Brain_tumor/Brain_tumor/set2/'
PATH_BRAIN_SET3_DICOM = PATH + 'Brain_tumor/Brain_tumor/set3_tumor/'
PATH_HEART_DICOM = PATH + 'heart_ct_dcm/'

if __name__ == '__main__':
    I_r, I_g, I_b = process_image_stack(path=PATH_TIF)

    ###################### test visualization #############################
    if test_case == 1:
        plot_RGB_samples(I_r, I_b, I_g)

    ###################### process brain set1 dicom #############################

    if test_case == 2.1:
        os.chdir(path=PATH_BRAIN_SET1_DICOM)
        dcm_dims, dcm_pixel_spacing, dcm_arr = process_dicom_stack(path=PATH_BRAIN_SET1_DICOM)
        plot_dicom_samples(dcm_dims, dcm_pixel_spacing, dcm_arr,
                           save_path=PATH_BRAIN_SET1_DICOM,
                           mode=mode,
                           num_of_images=num_samples)

    ###################### process brain set2 dicom #############################

    if test_case == 2.2:
        os.chdir(path=PATH_BRAIN_SET2_DICOM)
        dcm_dims, dcm_pixel_spacing, dcm_arr = process_dicom_stack(path=PATH_BRAIN_SET2_DICOM)
        plot_dicom_samples(dcm_dims, dcm_pixel_spacing, dcm_arr,
                           save_path=PATH_BRAIN_SET2_DICOM,
                           mode=mode,
                           num_of_images=num_samples)

    ###################### process brain set3 dicom #############################

    if test_case == 2.3:
        os.chdir(path=PATH_BRAIN_SET3_DICOM)
        dcm_dims, dcm_pixel_spacing, dcm_arr = process_dicom_stack(path=PATH_BRAIN_SET3_DICOM)
        plot_dicom_samples(dcm_dims, dcm_pixel_spacing, dcm_arr,
                           save_path=PATH_BRAIN_SET3_DICOM,
                           mode=mode,
                           num_of_images=num_samples)

    ###################### process heart dicom ###############################

    if test_case == 2.4:
        os.chdir(path=PATH_HEART_DICOM)
        dcm_dims, dcm_pixel_spacing, dcm_arr = process_dicom_stack(path=PATH_HEART_DICOM)
        plot_dicom_samples(dcm_dims, dcm_pixel_spacing, dcm_arr,
                           save_path=PATH_HEART_DICOM,
                           mode=mode,
                           num_of_images=num_samples)

    ###################### test neurons segmentation ############################

    if test_case == 3:
        os.chdir(path=PATH_TIF)
        seg_neuron_test_slice_by_slice(dir=PATH_TIF, dataset_name='Neuron Tiff Sequence')

    ###################### seg morph test ##################################

    if 4 < test_case < 5:
        if test_case == 4.1:
            dir = PATH_BRAIN_SET1_DICOM
        elif test_case == 4.2:
            dir = PATH_BRAIN_SET2_DICOM
        elif test_case == 4.3:
            dir = PATH_BRAIN_SET3_DICOM
        elif test_case == 4.4:
            dir = PATH_HEART_DICOM

        for filename in os.listdir(dir):
            if filename.endswith('.png'):
                img_path = dir + filename
                img = cv2.imread(img_path, 0)

                img, mask, eroded, dilated, opened, closed, grad = seg_brain_and_heart(img)

                plt.imshow(img, 'gray')
                plt.show()

                plot_images_general_case([mask, eroded, dilated, opened, closed, grad],
                                         6,
                                         (15, 10),
                                         titles=['binary',
                                                 'erosion',
                                                 'dilation',
                                                 'open morph',
                                                 'close morph',
                                                 'gradient'])


    ###################### choose threshold tests for each dataset ############

    if test_case == 5.1:
        thresh_test_slice_by_slice(PATH_BRAIN_SET1_DICOM, dataset_name='BRAIN SET 1')
    elif test_case == 5.2:
        thresh_test_slice_by_slice(PATH_BRAIN_SET2_DICOM, dataset_name='BRAIN SET 2')
    elif test_case == 5.3:
        thresh_test_slice_by_slice(PATH_BRAIN_SET3_DICOM, dataset_name='BRAIN SET 3')
    elif test_case == 5.4:
        thresh_test_slice_by_slice(PATH_HEART_DICOM, dataset_name='HEART SET')

    ###################### segmentation threshold adjust ui test ##############

    if test_case == 6.1:
        os.chdir(path=PATH_BRAIN_SET1_DICOM)
        threshold_ui(PATH_BRAIN_SET1_DICOM, percentile1=75, percentile2=100)
    elif test_case == 6.2:
        os.chdir(path=PATH_BRAIN_SET2_DICOM)
        threshold_ui(PATH_BRAIN_SET2_DICOM, percentile1=65, percentile2=85)
    elif test_case == 6.3:
        os.chdir(path=PATH_BRAIN_SET3_DICOM)
        threshold_ui(PATH_BRAIN_SET3_DICOM, percentile1=50, percentile2=75)
    elif test_case == 6.4:
        os.chdir(path=PATH_HEART_DICOM)
        threshold_ui(PATH_HEART_DICOM, percentile1=50, percentile2=80)
