import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5), # 6 is the output chanel size; 5 is the kernal size; 1 (chanel) 28 28 -> 6 24 24
            nn.MaxPool2d(2, 2),  # kernal size 2; stride size 2; 6 24 24 -> 6 12 12
            nn.ReLU(True),       # inplace=True means that it will modify the input directly thus save memory
            nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True) 
        )
        self.classifier =  nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x

# the function to test a neural network model using a test data loader
def testNN(model, testLoader, device):
    """
    Return a real number "accuracy" in [0,100] which counts 1 for each data instance; 
           a real number "singleAccuracy" in [0,100] which counts 1 for each number in the label 
    @param model: a PyTorch model whose accuracy is to be checked 
    @oaram testLoader: a PyTorch dataLoader object, including (input, output) pairs for model
    """
    # set up testing mode
    model.eval()

    # check if total prediction is correct
    correct = 0
    total = 0
    # check if each single prediction is correct
    singleCorrect = 0
    singleTotal = 0
    with torch.no_grad():
        for data, target in testLoader:
            output = model(data.to(device))
            if target.shape == output.shape[:-1]:
                pred = output.argmax(dim=-1) # get the index of the max value
            elif target.shape == output.shape:
                pred = (output >= 0).int()
            else:
                print(f'Error: none considered case for output with shape {output.shape} v.s. label with shape {target.shape}')
                import sys
                sys.exit()
            target = target.to(device).view_as(pred)
            correctionMatrix = (target.int() == pred.int()).view(target.shape[0], -1)
            correct += correctionMatrix.all(1).sum().item()
            total += target.shape[0]
            singleCorrect += correctionMatrix.sum().item()
            singleTotal += target.numel()
    accuracy = 100. * correct / total
    singleAccuracy = 100. * singleCorrect / singleTotal
    return accuracy, singleAccuracy