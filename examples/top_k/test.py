import sys
sys.path.append("../../")
import time

import torch

from dataGen import dataListTest, obsListTest
from network import FC
from neurasp import NeurASP

startTime = time.time()

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

nnRule = '''
nn(in(10, k), [true, false]).
'''

constraint = '''
% define maxweight k 
#const k = 7.

% we make a mistake if the total weight of the chosen items exceeds maxweight 
:- #sum{1, I : in(I,k,true)} > k.
'''

########
# Define nnMapping and initialze NeurASP object
########

m = FC(10, 50, 50, 50, 50, 50, 10)
nnMapping = {'in': m}
NeurASPobj = NeurASP(nnRule+constraint, nnMapping, optimizers=None)

########
# Load pretrained model
########

saveModelPath = 'data/model.pt'
m.load_state_dict(torch.load(saveModelPath, map_location='cpu'))

########
# Start testing
########
NeurASPobj.testConstraint(dataListTest, obsListTest,[constraint])
