# Most Reliable Path
Most Reliable Path is a variant of the [Shortest Path Problem](https://github.com/azreasoners/NeurASP/tree/master/examples/shortest_path), where, each edge is randomly associated with probabilities (0.512 or 0.8) which denotes the “reliability” of the edge, and the task is to find the most reliable path between the source and the destination node. We do not remove edges this time.

We use the same 5-layer MLP in the Shortest Path experiment as the baseline. We also use the simplepath and the reachability constraints to train the neural network by NeurASP. Besides, we use weak constraints to represent the probability of each edge in the grid.

## Train and Test
To start training, execute the following command under this folder
```
python train.py
```
the test accuracy will also be print out. 
