# Table tennis ball detection project

This project uses OpenCV and computer vision techniques to detect and track a table tennis ball.

Based on dataset https://www.kaggle.com/datasets/ketzoomer/table-tennis-ball-position-detection-dataset

Team members:<br>
    <li>Hanna Sholotiuk
    <li>Dmytro Broska 
    <li>Serhii Lukianchenko

It is not optimized for extended use. Instead, it showcases an idea of how OpenCV and computer vision can in a simple way be applied to analyze table tennis games.
     

1. [Overview](#overview)
2. [How to use](#usage)
</br>

## Overview

* **dataset_analysis** - Contains notebooks with scripts for analysing dataset

* **detection_and_tracking_methods** - Scripts that overview different methods of detection and tracking table tennis ball.

   * **CSRT_tracker_on_back_sub.py** - algorithm that uses the CSRT tracker in combination with background subtraction
   * **white_mask_tracking.py** - algorithm with mask selection for moving white objects
   * **ttnet-ball-detection.ipynb** - the implementation of two stages of the ball detection requested by the authors of the dataset
   * **optical_flow_farneback.py** - optical flow algorithm, that us the Farneback method
   * **optical_flow_pyrLK.py** - optical flow algorithm, that us the Lucas-Kanade method
   * **background_subtraction_contour_detection.py** - detection based on background subtraction with finding contours
   * **background_subtraction_blob_detection.py** - detection based on background subtraction with a simple blob detector

* **yolo_training_notebooks** - contains of notebooks with training scripts for different train videos from dataset

* **extended_yolo.py** - a script that uses a set of methods for tracking and detecting a ball. It consists of YOLOv5, CSRT tracker and background subtraction method

</br>

## Usage

### Step 1: Clone repository

Clone repository to desired location.

### Step 2: Run script

You can run script <b>extended_yolo.py</b> with video or sequence of images from test folder from dataset. Create a new folder called "videos" in the working directory and store videos or images here.

### Step 3: Analyze result

After running you can see the result of the detection and tracking table tennis ball
