import torch.nn as nn

######################################
# Structure of NN
# Instead of using a classification network to do the classification again on the bounding box
# we directly use the classification from Yolo
######################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    def forward(self, x):
        return x