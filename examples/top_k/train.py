import sys
sys.path.append("../../")
import time

import torch

from dataGen import dataList, obsList, dataListTest, obsListTest
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
# Define nnMapping and optimizers, initialze NeurASP object
########

m = FC(10, 50, 50, 50, 50, 50, 10)
nnMapping = {'in': m}
optimizer = {'in': torch.optim.Adam(m.parameters(), lr=0.001)}
NeurASPobj = NeurASP(nnRule+constraint, nnMapping, optimizer)

########
# Start training and testing
########
saveModelPath = 'data/model.pt'

for i in range(20):
    print('Epoch {}...'.format(i+1))
    time1 = time.time()
    NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=10, opt=True, smPickle='data/stableModels.pickle', bar=True)
    time2 = time.time()
    NeurASPobj.testConstraint(dataListTest, obsListTest,[constraint])
    print('--- train time: %s seconds ---' % (time2 - time1))
    print('--- test time: %s seconds ---' % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )

print('Storing the trained model into {}'.format(saveModelPath))
torch.save(m.state_dict(), saveModelPath)