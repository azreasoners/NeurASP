# Sudoku Solving
In the Sudoku Solving problem, we use a neural network to learn to solve Sudoku problems. The task is, given the textual representation of an unsolved Sudoku board (in the form of a 9 × 9 matrix where an empty cell is represented by 0), let a neural network learn to predict the solution of the Sudoku board.

## File Description
* data: a folder containing the training and testing data stored in pickle files
* train.py: a Python file that defines the NeurASP program and calls learn method in neurasp package.
* test.py: a Python file that defines the NeurASP program and calls testNN method in neurasp package.
* dataGen.py: a Python file that load the train/test data and generate dataList, obsList (for training), and test_loader (for testing)
* network.py: a Python file that defines the network "sol".
* solving_sudoku.ipynb: a Jupyter notebook for this example with detailed explanations to the codes

## Dataset and Pretrained Models
To make our NeurASP repository as small as possible, we put the dataset and our pre-trained models on dropbox and list their download links below. The dataset is stored in 2 pickle files where "easy_130k_given.p" stores the configurations of the given boards and "easy_130k_solved.p" stores their solutions. The pre-trained models are named as "model_epochX.pt" where X is the number of epochs taken for training.
* [easy_130k_given.p](https://www.dropbox.com/s/oyiwtchxdwyizcx/easy_130k_given.p?dl=1)
* [easy_130k_solved.p](https://www.dropbox.com/s/p51trvljuhfa4bq/easy_130k_solved.p?dl=1)
* [model_epoch10.pt](https://www.dropbox.com/s/pnebmwm3bstgdnz/model_epoch10.pt?dl=1)
* [model_epoch20.pt](https://www.dropbox.com/s/rul0frvh90rrgl3/model_epoch20.pt?dl=1)
* [model_epoch30.pt](https://www.dropbox.com/s/remjadql80epxnc/model_epoch30.pt?dl=1)
* [model_epoch40.pt](https://www.dropbox.com/s/hx073ahgdgtp55f/model_epoch40.pt?dl=1)
* [model_epoch50.pt](https://www.dropbox.com/s/u6ypv525oii6qxx/model_epoch50.pt?dl=1)
* [model_epoch60.pt](https://www.dropbox.com/s/nnadj2hfnptglrs/model_epoch60.pt?dl=1)
* [model_epoch70.pt](https://www.dropbox.com/s/hiugu38uu6wjnu4/model_epoch70.pt?dl=1)

If you want to test on this example, please download the dataset files and move them into the data folder. If you want to use our pretrained models, please also download them and move them into the data folder. 

## Train
To start training, execute the following command under this folder.
```
python train.py
```
The trained models will be saved in file "data/model_epochX.pt" where X is the epoch number. The test accuracy will also be printed out. 

## Test
You can use the following command to check the accuracy of a given pre-trained model. 
```
python test.py MODEL
```
where MODEL is the path to a pre-trained model. For example, you may download a pre-trained model above and put it under the data folder. Say the model is "data/model_epoch60.pt". Then you may use the following command to check the accuracy of "data/model_epoch60.pt" on the testing data.
```
python test.py data/model_epoch60.pt
```