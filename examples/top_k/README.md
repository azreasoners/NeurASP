# Tok-k
We consider a simple version of the knapsack problem, where each item is associated with a value and the task is to choose a subset of the items that maximizes the sum of the values of the items. We assume there are 10 items with the same weight 2, and the capacity of the knapsack is 15. For example,

    [2,7,3,5,2,3,8,2,1,5] [1,2,3,4,5,6,9]

is a labeled example such that the first list specifies the values of the 10 items and the second list is a solution that specifies the indices of the items to be put into the knapsack. Since the capacity of the knapsack is fixed to be 15 and each item has weight 2, one can infer that the solutions always contain 7 items.

## Train and Test
To start training, execute the following command under this folder
```
python train.py
```
the test accuracy will also be print out. 
