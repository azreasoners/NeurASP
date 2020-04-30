# Tok-k
We consider a simple version of the knapsack problem, where each item is associated with a value and the task is to choose a subset of the items that maximizes the sum of the values of the items. We assume there are 10 items with the same weight 2, and the capacity of the knapsack is 15. For example,

    [2,7,3,5,2,3,8,2,1,5] [1,2,3,4,5,6,9]

is a labeled example such that the first list specifies the values of the 10 items and the second list is a solution that specifies the indices of the items to be put into the knapsack. Since the capacity of the knapsack is fixed to be 15 and each item has weight 2, one can infer that the solutions always contain 7 items.

## File Description
* data: a folder containing the training and testing data in the file data.txt. Each row in data.txt contains 3 lists. The first list contains 10 numbers of possibly same values. The second list can be ignored since they represent the weight of each number. The third list is the label, contains the indices of the top 7 values.
* train.py: a Python file that defines the NeurASP program and calls learn method in neurasp package.
* test.py: a Python file that defines the NeurASP program and calls testConstraint method in neurasp package.
* dataGen.py: a Python file that load the train/test data and generate dataList, obsList (for training), and dataListTest, obsListTest (for testing)
* network.py: a Python file that defines the network "in".
* top_k.ipynb: a Jupyter notebook for this example with detailed explanations to the codes

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