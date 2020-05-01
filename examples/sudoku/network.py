from torch import nn
import torch.nn.functional as F

class Sudoku_Net(nn.Module):
    #add relu after 1x1conv add FC layer, dropout, adaptive pooling
    def __init__(self):
        super(Sudoku_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32,kernel_size=4,stride=2)
        self.conv1_bn=nn.BatchNorm2d(32)
        self.dropout1=nn.Dropout(p=.25)
        self.conv2 = nn.Conv2d(32, 64,kernel_size=3,stride=2)
        self.conv2_bn=nn.BatchNorm2d(64)
        self.dropout2=nn.Dropout(p=.25)
        self.conv3 = nn.Conv2d(64, 128,kernel_size=3,stride=2)
        self.conv3_bn=nn.BatchNorm2d(128)
        self.dropout3=nn.Dropout(p=.25)
        self.conv4 = nn.Conv2d(128, 256,kernel_size=2,stride=1)
        self.conv4_bn=nn.BatchNorm2d(256)
        self.dropout4=nn.Dropout(p=.25)
        self.conv5 = nn.Conv2d(256, 512,kernel_size=2,stride=1)
        self.conv5_bn=nn.BatchNorm2d(512)
        self.dropout5=nn.Dropout(p=.25)
        
        self.maxpool=nn.MaxPool2d(3)
        self.adaptive_avg_pool=nn.AdaptiveAvgPool2d((9,9))
        
        self.conv1x1_1=nn.Conv2d(in_channels=512,out_channels=10,kernel_size=1)
        self.conv1x1_2=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1)
        self.conv1x1_3=nn.Conv2d(in_channels=512,out_channels=10,kernel_size=1)
        
        self.fc1=nn.Linear(41472,81*10)
        self.dropout5=nn.Dropout(p=.25)
        
    def forward(self, x):
        x = self.dropout1(self.conv1_bn(self.conv1(x)))
        x = F.relu(x)
        x = self.dropout2(self.conv2_bn(self.conv2(x)))
        x = F.relu(x)
        x = self.dropout3(self.conv3_bn(self.conv3(x)))
        x = F.relu(x)
        x = self.dropout4(self.conv4_bn(self.conv4(x)))
        x = F.relu(x)
        x = self.dropout5(self.conv5_bn(self.conv5(x)))
        x = F.relu(x)
        x= self.maxpool(x)
        x=self.conv1x1_1(x)
        x=nn.Softmax(1)(x)
        batch_size=len(x)
        x=x.permute(0,2,3,1).contiguous().view(batch_size,810)
        x=x.view(batch_size,81,10)
        return x