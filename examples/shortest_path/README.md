# Shortest Path
Consider a 4*4 grid. There are 16 nodes and 24 edges in the grid. Given any 2 nodes in the grid, the target is to find a shortest path between them. We use the dataset from [(Xu et al. 2018)](https://arxiv.org/abs/1711.11157), which was used to demonstrate the effectiveness of semantic constraints for enhanced neural network learning. Each example is a 4 by 4 grid G = (V,E), where |V | = 16,|E| = 24. The source and the destination nodes are randomly picked up, as well as 8 edges are randomly removed to increase the difficulty. The dataset is divided into 60/20/20 train/validation/test examples.

We apply NeurASP on this problem using the same dataset and the neural network model from (Xu et al. 2018), but with a different training target: to maximize the probability of the training data under the semantics of NeurASP.

## Train and Test
To start training, execute the following command under this folder
```
python train.py
```
the test accuracy will also be print out. 
