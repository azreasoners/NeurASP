# MNIST Addition Baseline
The test accuracy will also be printed out by running either train_pureNN.py or train.py. Thus we do not provide a separate test file for baseline.

## File Description
* train_pureNN.py: a Python file that does usual NN training with cross entropy loss (i.e., nll_loss in PyTorch). 
* train.py: a Python file that defines the NeurASP program and calls learn method in neurasp package. This NeurASP program simulates pure NN training by using a single neural atom. 
* dataGen.py: a Python file that load the train/test data and generate dataList, obsList (for training), and test_loader (for testing)
* network.py: a Python file that defines the network "addition".

## Train with Usual Method
To start training with usual NN method, execute the following command under this folder.
```
python train_pureNN.py
```
The test accuracy will also be print out. 

## Simulate ure NN Training with NeurASP
You can also simulate pure NN training with NeurASP by using a NeurASP program with a single neural atom. You can execute the following command under this folder.
```
python train.py
```
The test accuracy will also be print out. 