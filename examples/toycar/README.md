# Toycar
This example shows how common-sense reasoning can be applied to neural network outputs in a single framework --- NeurASP. We prepared 2 images (used in the paper) in the data folder, each image contains objects such as cars and persons. The task is to figure it out which objects in the images are toys. Such task requires common-sense reasoning including default reasoning and recursive definition.

## File Description
* data: a folder containing 2 examples images used in the paper. Users may put other images in this folder to test.
* yolo: a folder including Yolo related files from https://github.com/eriklindernoren/PyTorch-YOLOv3 .
* infer.py: a Python file that defines the NeurASP program and calls inference method in neurasp package.
* dataGen.py: a python file that, for each image in data folder, 
  1. reshapes the image to 416 * 416; 
  2. feeds the reshaped image into Yolo network and apply non_max_suppression on raw output of Yolo to obtain a list of tuples (class, x1, y1, x2, y2, probability), each for one bounding box; 
  3. generates a string of ASP facts of the form "box(img, bi, x1, y1, x2, y2)." where img represent the reshaped image, bi is the identifier of a bounding box (in img) whose coordinates of left-top and right-bottom corners are (x1, y1) and (x2, y2); and 
  4. constructs a disctionary dataDic that maps each bi to a tensor of shape (1, 4) denoting the probabilites of classifying bi to the 4 classes: "person", "car", "truck", "other".
* network.py: a Python file that defines the network "label". Since Yolo already did classifications on bounding boxes with high accuracy, here we directly take the Yolo prediction as the final prediction.

## Inference and Interpreting the Result
To start inference on these 2 images, execute the following command under this folder.
```
python infer.py
```
For each image, we infer whether the objects in the image are toys or not by showing the most probable stable model of the NeurASP program. If the stable model contains, e.g., "toy(img,b1)", that means the bounding box b1 is a toy.