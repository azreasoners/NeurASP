# MNIST Addition
The digit addition problem is a simple but illustrative example used in [(Manhaeve et al. 2018)](https://arxiv.org/abs/1805.10872) to illustrate DeepProbLog’s ability to do both logical reasoning and deep learning. The task is, given a pair of digit images (MNIST) and their sum as the label, let a neural network learn the digit classification of the input images.

## File Description
* data: a folder containing the training and testing data, each data instance is a 3-tuple (idx1, idx2, sum)
* baseline: a folder containing the python files for the baseline
* train.py: a Python file that defines the NeurASP program and calls learn method in neurasp package.
* test.py: a Python file that defines the NeurASP program and calls testNN method in neurasp package.
* dataGen.py: a Python file that load the train/test data and generate dataList, obsList (for training), and test_loader (for testing)
* network.py: a Python file that defines the network "digit".
* mnist.ipynb: a Jupyter notebook for this example with detailed explanations to the codes

## Train
To start training, execute the following command under this folder.
```
python train.py
```
The trained model will be saved in file "data/model.pt" and the test accuracy will also be printed out. 

## Test
If you already generated the trained model "data/model.pt", you can directly check the accuracy on the testing data by executing the following command under this folder.
```
python test.py
```