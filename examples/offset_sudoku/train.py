import torch
import torch.optim as optim

from network import Sudoku_Net_Offset_bn
from dataGen import train_loader,validation_loader,to_onehot
from Trainer import Train_Test

# =============================================================================
# Instantiate Network
# =============================================================================

model = Sudoku_Net_Offset_bn()
model.cuda()

# =============================================================================
# Network Parameters
# =============================================================================

learning_rate = .0001
epochs = 2000

opt = optim.Adam(model.parameters(),lr=learning_rate)
criterion = torch.nn.BCELoss()

# =============================================================================
# Training
# =============================================================================

model = Train_Test(model, train_loader, validation_loader, opt, criterion, epochs)
torch.save(model.state_dict(), 'model_data70.pt')
