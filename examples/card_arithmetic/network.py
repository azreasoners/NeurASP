import torch


class CardNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize a neural network for processing playing card images
        """
        super(CardNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, 3)
        self.conv2 = torch.nn.Conv2d(64, 64, 3)
        self.conv3 = torch.nn.Conv2d(64, 128, 3)
        self.conv4 = torch.nn.Conv2d(128, 128, 3)
        self.fc1 = torch.nn.Linear(17280, 512)
        self.fc2 = torch.nn.Linear(512, 52)

        self.pool = torch.nn.MaxPool2d(2, 2)
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.5)
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(1)


    def forward(self, x):
        """
        Process the input image
        :param x: Batched input images
        :return: Prediction of downstream label as a logit tensor
        """
        x = x.flatten(0, 1)
        x = self.pool(self.ReLU(self.conv1(x)))
        x = self.pool(self.ReLU(self.conv2(x)))
        x = self.pool(self.ReLU(self.conv3(x)))
        x = self.pool(self.ReLU(self.conv4(x)))

        x = self.flatten(x)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
