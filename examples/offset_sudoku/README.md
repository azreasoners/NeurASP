# Offset Sudoku Perception

For this task we use consider an offset Sudoku board for perception which is in the form of an RBG image. The base neural network is adjusted slightly to account for 3 color channels, but overall this architecture is similar to the regular sudoku image perception. 

## File Description
* data: a folder contraining 2 pickle files (storing the Sudoku images and their labels) and their examples.
* train.py: a Python file for baseline training and testing
* test.py: a Python file for testing a given PyTorch model. 
* infer.py: a Python file that defines the NeurASP program and infer the solution of a given offset Sudoku image using both perception and reasoning.
* trainer.py: a Python file which includes the function to train a model. 
* dataGen.py: a Python file that loads the training and test data // edit for dataList etc.
* network.py: a Python file that defines the network "identify".
* offset_perception_noteobok.pynb: a Jupyter notebook for training the baseline.

## Train
To start training, execute the following command under this folder.
```
python train.py 
```

## Test
To test, make sure the path to your model is included in test.py, and run the following command:
```
python test.py
```

## Inference on Single Sudoku Image
Given an offset Sudoku image, you can use the following command to infer its solution
```
python infer.py IMAGE
```
where MAGE must be the path to an offset Sudoku image. For example, you can try the following command to infer the solution of our example normal Sudoku problem in "data/offset_sudoku.png".
```
python infer.py data/offset_sudoku.png
```
Note that this script assumes the existence of a pre-trained model "data/model_data70.pt". You may train such model by running
```
python train.py
```