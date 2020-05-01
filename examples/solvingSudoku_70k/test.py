import sys
sys.path.append('../../')
import time

import torch

from dataGen import train_loader, test_loader
from neurasp import NeurASP
from network import Sudoku_Net

startTime = time.time()

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram = '''
% neural rule
nn(sol(81, config), [1,2,3,4,5,6,7,8,9]).

% we assign one number at each position (R,C)
a(R,C,N) :- sol(Pos, config, N), R=Pos/9, C=Pos\9.

% it's a mistake if the same number shows 2 times in a row
:- a(R,C1,N), a(R,C2,N), C1!=C2.

% it's a mistake if the same number shows 2 times in a column
:- a(R1,C,N), a(R2,C,N), R1!=R2.

% it's a mistake if the same number shows 2 times in a 3*3 grid
:- a(R,C,N), a(R1,C1,N), R!=R1, C!=C1, ((R/3)*3 + C/3) = ((R1/3)*3 + C1/3).
'''

########
# Define nnMapping and initialze NeurASP object
########

m = Sudoku_Net()
nnMapping = {'sol': m}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers=None)

########
# Load pretrained model
########

try:
	saveModelPath = sys.argv[1]
except:
	saveModelPath = 'data/model_epoch60.pt'
print('Loading the trained model from {}'.format(saveModelPath))
m.load_state_dict(torch.load(saveModelPath, map_location='cpu'))

########
# Start testing
########
print('\nNote that the following accuracy are obtained without the use of the inference trick\n')
acc, singleAcc = NeurASPobj.testNN('predict', test_loader)
print('Test Acc (whole board): {:0.2f}%'.format(acc))
print('Test Acc (single cell): {:0.2f}%'.format(singleAcc))
