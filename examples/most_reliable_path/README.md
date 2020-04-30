# Most Reliable Path
Most Reliable Path is a variant of the [Shortest Path Problem](https://github.com/azreasoners/NeurASP/tree/master/examples/shortest_path), where, each edge is randomly associated with probabilities (0.512 or 0.8) which denotes the “reliability” of the edge, and the task is to find the most reliable path between the source and the destination node. We do not remove edges this time.

We use the same 5-layer MLP in the Shortest Path experiment as the baseline. We also use the simplepath and the reachability constraints to train the neural network by NeurASP. Besides, we use weak constraints to represent the probability of each edge in the grid.

## File Description
* data: a folder containing the training and testing data in the file data.txt. Each row in data.txt is a vector of length (2+24+24=50). The first 2 numbers are the indices of the start and end nodes, the following 24 probabilities are the probabilities for 24 edges, and the last 24 numbers form the label where 1 means the edge should be in the most reliable path.
* train.py: a Python file that defines the NeurASP program and calls learn method in neurasp package.
* test.py: a Python file that defines the NeurASP program and calls testConstraint method in neurasp package.
* dataGen.py: a Python file that load the train/test data and generate dataList, obsList (for training), and dataListTest, obsListTest (for testing)
* network.py: a Python file that defines the network "mrp".
* most_reliable_path.ipynb: a Jupyter notebook for this example with detailed explanations to the codes

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