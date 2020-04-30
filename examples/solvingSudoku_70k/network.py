from torch import nn
import torch.nn.functional as F

class Sudoku_Net(nn.Module):
    def __init__(self):
        super(Sudoku_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 512,kernel_size=3,stride=1,padding=1)        
        self.conv2=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)        
        self.conv3=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)        
        self.conv4=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)        
        self.conv5=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)        
        self.conv6=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)        
        self.conv7=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)        
        self.conv8=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)        
        self.conv9=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)        
        self.conv1x1=nn.Conv2d(in_channels=512,out_channels=9,kernel_size=1)
        
    def forward(self, x_orig):        
        x=self.conv1(x_orig)
        x=F.relu(x)
        
        x=self.conv2(x)
        x=F.relu(x)
        
        x=self.conv3(x)
        x=F.relu(x)
        
        x=self.conv4(x)
        x=F.relu(x)
        
        x=self.conv5(x)
        x=F.relu(x)
        
        x=self.conv6(x)
        x=F.relu(x)
        
        x=self.conv7(x)
        x=F.relu(x)
        
        x=self.conv8(x)
        x=F.relu(x)
        
        x=self.conv9(x)
        x=F.relu(x)

        x=self.conv1x1(x)
        x=x.permute(0,2,3,1)
        x=x.view(-1,81,9)
        x=nn.Softmax(2)(x)
        return x