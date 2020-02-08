# NeurASP
NeurASP: Neural Networks + Answer Set Programs

# Introduction
NeurASP is a simple extension of answer set programs by embracing neural networks. By treating the neural network output as the probability distribution over atomic facts in answer set programs, NeurASP provides a simple and effective way to integrate sub-symbolic and symbolic computation. This repository includes examples to show
1. how NeurASP can make use of pretrained neural networks in symbolic computation and how it can improve the perception accuracy of a neural network by applying symbolic reasoning in answer set programming; and
2. how NeurASP is used to train a neural network better by training with rules so that a neural network not only learns from implicit correlations from the data but also from the explicit complex semantic constraints expressed by ASP rules.

## Prerequisite
1. Install either Anaconda or Miniconda according to the [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. Install [`clingo`](https://potassco.org/clingo/) via
```
conda install -c potassco clingo
```
3. Install PyTorch according to the [link](https://pytorch.org/).

## Installation
Clone this repo:
```
git clone https://github.com/azreasoners/NeurASP
cd NeurASP
```
