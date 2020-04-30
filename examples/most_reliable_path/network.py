from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random
import numpy as np


seed_num = 1
random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True

class FC(nn.Module):

    def __init__(self, *sizes):
        super(FC, self).__init__()
        layers = []
        print("Neural Network (MLP) Structure: {}".format(sizes))
        for i in range(len(sizes)-2):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        layers.append(nn.Sigmoid())
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)