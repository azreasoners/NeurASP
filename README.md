# NeurASP
NeurASP: Neural Networks + Answer Set Programs

# Introduction
NeurASP is a simple extension of answer set programs by embracing neural networks. By treating the neural network output as the probability distribution over atomic facts in answer set programs, NeurASP provides a simple and effective way to integrate sub-symbolic and symbolic computation. This repository includes examples to show
1. how NeurASP can make use of pretrained neural networks in symbolic computation and how it can improve the perception accuracy of a neural network by applying symbolic reasoning in answer set programming; and
2. how NeurASP is used to train a neural network better by training with rules so that a neural network not only learns from implicit correlations from the data but also from the explicit complex semantic constraints expressed by ASP rules.

## Installation
0. We assume Anaconda is installed. One can install it according to its [installation page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
1. Clone this repo:
```
git clone https://github.com/azreasoners/NeurASP
cd NeurASP
```
2. Create a virtual environment `neurasp`. Install clingo (ASP solver) and tqdm (progress meter).
```
conda create --name neurasp python=3.9
conda activate neurasp
conda install -c potassco clingo=5.5 tqdm
```
3. Install Pytorch according to its [Get-Started page](https://pytorch.org/get-started/locally/). Below is an example command we used on Linux with cuda 10.2. (PyTorch version 1.12.0 is tested.)
```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## Examples
We provide 3 inference and 5 learning examples as shown below. Each example is stored in a separate folder with a readme file.
### Inference Examples
* [Sudoku](https://github.com/azreasoners/NeurASP/tree/master/examples/sudoku)
* [Offset Sudoku](https://github.com/azreasoners/NeurASP/tree/master/examples/offset_sudoku)
* [Toy-car](https://github.com/azreasoners/NeurASP/tree/master/examples/toycar)

### Learning Examples
* [MNIST Addition](https://github.com/azreasoners/NeurASP/tree/master/examples/mnistAdd)
* [Shorstest Path](https://github.com/azreasoners/NeurASP/tree/master/examples/shortest_path)
* [Sudoku Solving](https://github.com/azreasoners/NeurASP/tree/master/examples/solvingSudoku_70k)
* [Top-k](https://github.com/azreasoners/NeurASP/tree/master/examples/top_k)
* [Most Reliable Path](https://github.com/azreasoners/NeurASP/tree/master/examples/most_reliable_path)

## Related Work
You may also be interested in our work [Injecting Logical Constraints into Neural Networks via Straight-Through-Estimators](http://peace.eas.asu.edu/joolee/papers/ste-ns-icml.pdf). Its codes are available [here](https://github.com/azreasoners/cl-ste).