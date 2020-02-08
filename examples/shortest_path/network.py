import torch.nn as nn

######################################
# Structure of NN
######################################

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