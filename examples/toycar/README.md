# Toycar
This example shows how common-sense reasoning can be applied to neural network outputs in a single framework --- NeurASP. We prepared 2 images (used in the paper) in the data folder, each image contains objects such as cars and persons. The task is to figure it out which objects in the images are toys. Such task requires common-sense reasoning including default reasoning and recursive definition.

## File Description
* data: a folder containing 2 examples images used in the paper. Users may put other images in this folder to test.
* yolo: a folder including Yolo related files from https://github.com/eriklindernoren/PyTorch-YOLOv3 .
* dataGen.py: a python file that (i) defines dataList using the images in the data folder, and (ii) defines postProcessing function to apply non_max_suppression on raw output of Yolo and return a list of tuples (class, x1, y1, x2, y2, probability) that will be used by the neurasp pacakge.
* infer.py: a Python file that defines the NeurASP program and calls inference method in neurasp package.

## Inference and Interpreting the Result
To start inference on these 2 images, execute the following command under this folder.
```
python infer.py
```
For each image, we infer whether the objects in the image are toys or not by showing the most probable stable model of the NeurASP program. If the stable model contains, e.g., "toy(img,b1)", that means the bounding box b1 is a toy.