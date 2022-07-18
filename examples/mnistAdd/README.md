# MNIST Addition
The digit addition problem `mnistAdd` is a simple but illustrative example used in [(Manhaeve et al. 2018)](https://arxiv.org/abs/1805.10872) to illustrate DeepProbLog’s ability to do both logical reasoning and deep learning. The task is, given a pair of digit images (MNIST) and their sum as the label, let a neural network learn the digit classification of the input images.

This problem can be extended to `mnistAdd2` where each number consists of 2 digit images.

## File Description
* data: a folder containing the training and testing data, each data instance is a 3-tuple (idx1, idx2, sum)
* baseline: a folder containing the python files for the baseline
* mnistAdd.py: the Python script for `mnistAdd` example
* mnistAdd2.py: the Python script for `mnistAdd2` example
* dataGen.py: a Python file that load the train/test data and generate dataList, obsList (for training), and test_loader (for testing)
* network.py: a Python file that defines the network of digit classifier.
* mnist.ipynb: a Jupyter notebook for `mnistAdd` example with detailed explanations to the codes

## How to run
For `mnistAdd` problem
```
python mnistAdd.py
```
For `mnistAdd2` problem
```
python mnistAdd2.py
```