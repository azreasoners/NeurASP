# Sudoku Solving
In the Sudoku Solving problem, we use a neural network to learn to solve Sudoku problems. The task is, given the textual representation of an unsolved Sudoku board (in the form of a 9 × 9 matrix where an empty cell is represented by 0), let a neural network learn to predict the solution of the Sudoku board.

## File Description
* data: a folder containing the training and testing data, as well as 4 trained models
* train.py: a Python file that defines the NeurASP program and calls learn method in neurasp package.
* test.py: a Python file that defines the NeurASP program and calls testNN method in neurasp package.
* dataGen.py: a Python file that load the train/test data and generate dataList, obsList (for training), and test_loader (for testing)
* network.py: a Python file that defines the network "sol".
* solving_sudoku.ipynb: a Jupyter notebook for this example with detailed explanations to the codes

## Train
To start training, execute the following command under this folder.
```
python train.py
```
The trained models will be saved in file "data/model_epochX.pt" where X is the epoch number. The test accuracy will also be printed out. 

## Test
We provided a few trained models in the data folder. You can directly check the accuracy of "data/model_epoch60.pt" on the testing data by executing the following command under this folder.
```
python test.py
```