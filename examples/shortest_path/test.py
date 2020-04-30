import sys
sys.path.append("../../")

import torch

from dataGen import dataListTest, obsListTest
from network import FC
from neurasp import NeurASP

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

nnRule = '''
nn(sp(24, g), [true, false]).
'''

aspRule = '''
sp(X) :- sp(X,g,true).

sp(0,1) :- sp(0).
sp(1,2) :- sp(1).
sp(2,3) :- sp(2).
sp(4,5) :- sp(3).
sp(5,6) :- sp(4).
sp(6,7) :- sp(5).
sp(8,9) :- sp(6).
sp(9,10) :- sp(7).
sp(10,11) :- sp(8).
sp(12,13) :- sp(9).
sp(13,14) :- sp(10).
sp(14,15) :- sp(11).
sp(0,4) :- sp(12).
sp(4,8) :- sp(13).
sp(8,12) :- sp(14).
sp(1,5) :- sp(15).
sp(5,9) :- sp(16).
sp(9,13) :- sp(17).
sp(2,6) :- sp(18).
sp(6,10) :- sp(19).
sp(10,14) :- sp(20).
sp(3,7) :- sp(21).
sp(7,11) :- sp(22).
sp(11,15) :- sp(23).

sp(X,Y) :- sp(Y,X).
'''

constraints = {}

constraints['nr'] = '''
% [nr] 1. No removed edges should be predicted
mistake :- sp(X), removed(X).
'''

constraints['p'] = '''
% [p] 2. Prediction must form simple path(s)
% that is: the degree of nodes should be either 0 or 2
mistake :- X=0..15, #count{Y: sp(X,Y)} = 1.
mistake :- X=0..15, #count{Y: sp(X,Y)} >= 3.
'''

constraints['r'] = '''
% [r] 3. Every 2 nodes in the prediction must be reachable
reachable(X, Y) :- sp(X, Y).
reachable(X, Y) :- reachable(X, Z), sp(Z, Y).
mistake :- sp(X, _), sp(Y, _), not reachable(X, Y).
'''

constraints['o'] = '''
% [o] 4. Predicted path should contain least edges
:~ sp(X). [1, X]
'''

########
# Set up the list of constraint combinations for testing accuracy
########

combinations = [['nr'], ['p'], ['r'], ['nr', 'p'], ['nr', 'r'], ['p', 'r'], ['nr', 'p', 'r'], ['nr', 'p', 'r', 'o']]
combinations = [aspRule + ''.join([constraints[c] for c in combination]) for combination in combinations]

########
# Define nnMapping and initialze NeurASP object
########

m = FC(40, 50, 50, 50, 50, 50, 24)
nnMapping = {'sp': m}
NeurASPobj = NeurASP(nnRule, nnMapping, optimizers=None)

########
# Load pretrained model
########

saveModelPath = 'data/model.pt'
m.load_state_dict(torch.load(saveModelPath, map_location='cpu'))

########
# Start testing
########
NeurASPobj.testConstraint(dataList=dataListTest, obsList=obsListTest, mvppList=combinations)
