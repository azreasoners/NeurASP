# Toycar
This example shows how common-sense reasoning can be applied to neural network outputs in a single framework --- NeurASP. We prepared 2 images (used in the paper) in the data folder, each image contains objects such as cars and persons. The task is to figure it out which objects in the images are toys. Such task requires common-sense reasoning including default reasoning and recursive definition.

## File Description
* data: a folder containing 2 examples images used in the paper. Users may put other images in this folder to test.
* yolo: a folder containing Yolo related files from https://github.com/eriklindernoren/PyTorch-YOLOv3 .
* infer.py: a Python file that defines the NeurASP program and calls inference method in neurasp package.
* dataGen.py: a python file that defines helper functions. Given a domain (e.g., ["car", "cat", "person", "truck", "other"]), it mainly does the following steps for each image in data folder.
  1. Reshape the image to 416 * 416.
  2. Feed the reshaped image into Yolo network and apply non_max_suppression on raw output of Yolo to obtain a list of tuples (class, x1, y1, x2, y2, probability), each for one bounding box.
  3. Generate a string of ASP facts of the form "box(img, bi, x1, y1, x2, y2)." where img represent the reshaped image, bi is the identifier of a bounding box (in img) whose coordinates of left-top and right-bottom corners are (x1, y1) and (x2, y2).
  4. Construct a disctionary dataDic that maps each bi to a tensor of shape (1, len(domain)) denoting the probabilites of classifying bi to each class in the given domain.
* network.py: a Python file that defines the network "label". Since Yolo already did classifications on bounding boxes with high accuracy, here we directly take the Yolo prediction as the final prediction.

## Pretrained Model
To make our NeurASP repository as small as possible, we put the pre-trained Yolo model on dropbox and list its download link below. To infer the relationships among objects in an image, you need to first download this model and move it into the yolo folder.
* [yolov3.weights](https://www.dropbox.com/s/qakg1tw1hgd805e/yolov3.weights?dl=1)

## Inference and Interpreting the Result
To start inference on these 2 images, execute the following command under this folder.
```
python infer.py
```
For each image, we infer whether the objects in the image are toys or not by showing the most probable stable model of the NeurASP program. If the stable model contains, e.g., "toy(img,b1)", that means the bounding box b1 is a toy.