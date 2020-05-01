import torch
import torch.optim as optim

from network import Sudoku_Net_Offset_bn
from dataGen import train_loader,validation_loader,to_onehot
from Trainer import Test

# =============================================================================
# Instantiate Network
# =============================================================================

model = Sudoku_Net_Offset_bn().cuda()
model.load_state_dict(torch.load('model_name'))
# =============================================================================
# Network Parameters
# =============================================================================

learning_rate=.0001
epochs=2000

opt=optim.Adam(model.parameters(),lr=learning_rate)
criterion=torch.nn.BCELoss()

# =============================================================================
# Testing
# =============================================================================





model=Test(model, train_loader, validation_loader, opt, criterion, epochs)

 
