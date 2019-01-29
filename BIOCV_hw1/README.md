## Homework 1: Image Segmentation

__by Jiarong Ye (jxy225)__

OS:
- Ubuntu 18.04

Python Version:
- python 3.6.2

OpenCV-python Version:
- 3.4.2

DataSet storage path:
- your-dataset-path
    - Neuron_TIff_sequence
        - 02_050050.tif
        - 02_050051.tif
        - 02_050052.tif
        - ...
        - ...
        - 02_050130.tif
    - Brain_tumor
        - Brain_tumor
            - set1
                - 119_11.8016.9.1137006610104474270801.1.1.1.1.0.0.2.dcm
                - 120_1.8016.9.1137006610104474270801.10.1.1.1.0.0.2.dcm
                - ...
                - ...
            - set2
                - 95_11.8016.9.1137006610104474270701.1.1.1.1.0.0.1.dcm
                - 96_1.8016.9.1137006610104474270701.10.1.1.1.0.0.1.dcm
                - ...
                - ...
                - 118_11.8016.9.1137006610104474270701.9.1.1.1.0.0.1.dcm
            - set3_tumor
                - 143_11.8016.9.1137006610104474270901.1.1.1.1.0.0.1.dcm
                - 144_1.8016.9.1137006610104474270901.10.1.1.1.0.0.1.dcm
                - ...
                - ...
                - 166_11.8016.9.1137006610104474270901.9.1.1.1.0.0.1.dcm
    - heart_ct_dcm
        - 124
        - 125
        - ...
        - ...
        - 283


Basic Idea:

- Reading and Processing
    - For the neuron images:
        - First, split each image into R, G, B 3 channels and display slice by slice;
        ```python
        python3 main.py --path <your-dataset-path> --testcase 1
        ``` 
        - then extract the blue channel of the images
    - For the Brain and Heart CT scanned images which are in dicom format:
        - First, convert images in each set from dicom format to png
        ```python
        python3 main.py --path <your-dataset-path> --testcase 2.1 --mode 'display' --samples <num-of-images-you-want-to-display>
        python3 main.py --path <your-dataset-path> --testcase 2.2 --mode 'display' --samples <num-of-images-you-want-to-display>
        python3 main.py --path <your-dataset-path> --testcase 2.3 --mode 'display' --samples <num-of-images-you-want-to-display>
        python3 main.py --path <your-dataset-path> --testcase 2.4 --mode 'display' --samples <num-of-images-you-want-to-display>
        ``` 
        - Then save
        ```python
        python3 main.py --path <your-dataset-path> --testcase 2.1 --mode 'save'
        python3 main.py --path <your-dataset-path> --testcase 2.2 --mode 'save' 
        python3 main.py --path <your-dataset-path> --testcase 2.3 --mode 'save' 
        python3 main.py --path <your-dataset-path> --testcase 2.4 --mode 'save' 
        ``` 
- Segmentation of Neuron TIF sequence
    - First make sure the input image is extracted from blue channel
    - Apply gaussian blur on the image
    - Use thresholding to generate binary image
        - Default: OTSU thresholding
        - However, in the display GUI, the original and segmented images are displayed slice by slice, thus when there's 
        some among all images that you found the segmented result is suboptimal, you can:
            - press __'p' key__ on your keyboard to pause the loop
            - then manually adjust threshold with the scrollbars appear at the bottom 
            - when finished, press __'n' key__ on your keyboard to move on to the next slide
        ```python
        python3 main.py --path <your-dataset-path> --testcase 3
        ``` 
- Segmentation of Brain/Heart CT
    - 1: 
        - Morphological Operations for each set
            ```python
            # brain set 1
            python3 main.py --path <your-dataset-path> --testcase 4.1
            # brain set 2
            python3 main.py --path <your-dataset-path> --testcase 4.2
            # brain set 3
            python3 main.py --path <your-dataset-path> --testcase 4.3
            
            # heart set 
            python3 main.py --path <your-dataset-path> --testcase 4.4
            ``` 
    - 2 (better):
        - Connected Component Analysis for each set
            - First applying equalHist on input image for contrast enhancement before generating the mask 
            - Then get the histogram of input image, convolve the histogram with a simple box filter for smoothing
            - Next, extract the peaks from the histogram, sort them in the reverse order, get the corresponding indexes of these peaks
            - Calculate the percentiles of the peak indexes to test for the optimal lower and upper threshold
              ```python
                # brain set 1
                python3 main.py --path <your-dataset-path> --testcase 5.1
                # brain set 2
                python main.py --path <your-dataset-path> --testcase 5.2
                # brain set 3
                python3 main.py --path <your-dataset-path> --testcase 5.3
                
                # heart set 
                python3 main.py --path <your-dataset-path> --testcase 5.4
              ``` 
        - Final Result:
            - in the display GUI, the original and segmented images are displayed slice by slice, thus when there's 
        some among all images that you found the segmented result is suboptimal, you can:
            - press __'p' key__ on your keyboard to pause the loop
                - first do not move the scrollbar, see if it auto-adjusts (because in this part, I have a few very simple rules written 
                up for threshold adjustment when the approach based on histogram peaks percentile does not work well)
                - if the segmentation is still not good, then manually adjust threshold with the scrollbars appear at the bottom 
            - when finished, press __'n' key__ on your keyboard to move on to the next slide
            ```python
              # brain set 1
              python3 main.py --path <your-dataset-path> --testcase 6.1
              # brain set 2
              python3 main.py --path <your-dataset-path> --testcase 6.2
              # brain set 3
              python3 main.py --path <your-dataset-path> --testcase 6.3
                    
              # heart set 
              python3 main.py --path <your-dataset-path> --testcase 6.4
            ``` 
    
    - Testing all
        - please see the testing videos __test.mp4__ of each test mentioned above    
            
      
        
   