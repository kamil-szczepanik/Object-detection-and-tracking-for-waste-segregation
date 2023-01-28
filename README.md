# Object detection and tracking for waste segregation

This repository is an attempt to create a real-time computer vision system that solves multiple object detection and tracking task.

## Dataset
[ZeroWaste](https://github.com/dbash/zerowaste/) has been used to train instance segmentation models. Frames from test dataset were used to put together videos for testing tracking algorithms.

## Instance segmentation models and detection
Detection part in this project is actually done by instance segmentation architecutres. Although masks haven't been used so far, it's possible to use it in some tracking algorithms in the future. Only bounding boxes have been used in the project. **For this task Mask R-CNN was used.**


 - The original idea was also to use YOLACT++ model but it's performance was significantly worse than Mask R-CNN. 
 - In this repo there are provided Colab Notebooks with training and evaluation processes of each model: [Mask R-CNN](detection/maskrcnn/), [YOLACT++](detection/yolact++/).
- To reproduce results presented in those notebooks a full ZeroWaste dataset is necessary. 
#### Mask R-CNN
Implementation from [Detectron2](https://github.com/facebookresearch/detectron2) repo.
#### YOLACT++
Implementation from [yolact](https://github.com/dbolya/yolact) repo.
 
 ## Tracking algorihtms
 
 ### SORT
 Implementation from [sort](https://github.com/abewley/sort) repo
 
 ### DeepSORT
  Implementation from [deep_sort_realtime](https://github.com/levan92/deep_sort_realtime) repo
  
 ### KCF
Multitracker created using OpenCV KCFTracker class.

## Quickstart in Google Colab
Whole project was done in Google Colab environment. This [Colab Notebook](https://colab.research.google.com/drive/1SA4f0LhkRf6HNSk1X6s-yK0NZI2ly2Ex#scrollTo=2iLf2_sVnH4F) you'll find examples that show how to start the program

