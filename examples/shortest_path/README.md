# Shortest Path
Consider a 4*4 grid. There are 16 nodes and 24 edges in the grid. Given any 2 nodes in the grid, the target is to find a shortest path between them. We use the dataset from [(Xu et al. 2018)](https://arxiv.org/abs/1711.11157), which was used to demonstrate the effectiveness of semantic constraints for enhanced neural network learning. Each example is a 4 by 4 grid G = (V,E), where |V | = 16,|E| = 24. The source and the destination nodes are randomly picked up, as well as 8 edges are randomly removed to increase the difficulty. The dataset is divided into 60/20/20 train/validation/test examples.

We apply NeurASP on this problem using the same dataset and the neural network model from (Xu et al. 2018), but with a different training target: to maximize the probability of the training data under the semantics of NeurASP.

## File Description
* data: a folder containing the training and testing data. The data is stored in the file shortestPath.data, which is from (Xu et al. 2018). Each row in shortestPath.data is a 3-tuple: the first element contains a sequence of 8 edge indices that are removed, the second element contains a sequence of 2 node indices that are start/end nodes, and the third element is the label containing a sequence of edge indices that should be in the shortest path.
* train.py: a Python file that defines the NeurASP program and calls learn method in neurasp package.
* test.py: a Python file that defines the NeurASP program and calls testConstraint method in neurasp package.
* dataGen.py: a Python file that load the train/test data and generate dataList, obsList (for training), and dataListTest, obsListTest (for testing)
* network.py: a Python file that defines the network "sp".
* shortest_path.ipynb: a Jupyter notebook for this example with detailed explanations to the codes

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