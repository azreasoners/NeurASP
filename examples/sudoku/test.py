import sys
sys.path.append('../../')

import torch

from dataGen import dataListTest, obsListTest, test_loader
from neurasp import NeurASP
from network import Sudoku_Net

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
######################################

dprogram = '''
% neural rule
nn(identify(81, img), [empty,1,2,3,4,5,6,7,8,9]).

% we assign one number at each position (R,C)
a(R,C,N) :- identify(Pos, img, N), R=Pos/9, C=Pos\9, N!=empty.
{a(R,C,N): N=1..9}=1 :- identify(Pos, img, empty), R=Pos/9, C=Pos\9.

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
nnMapping = {'identify': m}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers=None)

########
# Load pre-trained model and start testing
########

numOfData = [15, 17, 19, 21, 23, 25]

for num in numOfData:
    print('\nLoad the model trained with {} data'.format(num))
    m.load_state_dict(torch.load('data/model_data{}.pt'.format(num), map_location='cpu'))

    # start testing
    acc, singleAcc = NeurASPobj.testNN('identify', test_loader)
    print('Test Acc Using Pure NN (whole board): {:0.2f}%'.format(acc))
    print('Test Acc Using Pure NN (single cell): {:0.2f}%'.format(singleAcc))
    acc = NeurASPobj.testInferenceResults(dataListTest, obsListTest)
    print('Test Acc Using NeurASP (whole board): {:0.2f}%'.format(acc))