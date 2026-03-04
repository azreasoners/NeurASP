import os
import sys
sys.path.append('../../')
import time

import torch

from dataGen import dataList, obsList, testLoader
from network import Net
from neurasp import NeurASP

startTime = time.time()

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram = '''
nn(op(4,i), [0,1,2]).

apply(D1,O1,D2,O2,D3,R) :- digits(D1,D2,D3), O1=0, O2=0, R=(D1+D2)+D3.
apply(D1,O1,D2,O2,D3,R) :- digits(D1,D2,D3), O1=0, O2=1, R=(D1+D2)-D3.
apply(D1,O1,D2,O2,D3,R) :- digits(D1,D2,D3), O1=0, O2=2, R=(D1+D2)*D3.

apply(D1,O1,D2,O2,D3,R) :- digits(D1,D2,D3), O1=1, O2=0, R=(D1-D2)+D3.
apply(D1,O1,D2,O2,D3,R) :- digits(D1,D2,D3), O1=1, O2=1, R=(D1-D2)-D3.
apply(D1,O1,D2,O2,D3,R) :- digits(D1,D2,D3), O1=1, O2=2, R=(D1-D2)*D3.

apply(D1,O1,D2,O2,D3,R) :- digits(D1,D2,D3), O1=2, O2=0, R=(D1*D2)+D3.
apply(D1,O1,D2,O2,D3,R) :- digits(D1,D2,D3), O1=2, O2=1, R=(D1*D2)-D3.
apply(D1,O1,D2,O2,D3,R) :- digits(D1,D2,D3), O1=2, O2=2, R=(D1*D2)*D3.

apply2x2(D1,D2,D3,R1,R2,C1,C2) :- digits(D1,D2,D3), 
                                  op(0,i,O1), op(1,i,O2), op(2,i,O3), op(3,i,O4), 
                                  apply(D1,O1,D2,O2,D3,R1),
                                  apply(D1,O3,D2,O4,D3,R2),
                                  apply(D1,O1,D2,O3,D3,C1),
                                  apply(D1,O2,D2,O4,D3,C2).

'''

########
# Define nnMapping and optimizers, initialze NeurASP object
########

m = Net()
nnMapping = {'op': m}
optimizers = {'op': torch.optim.Adam(m.parameters(), lr=0.001)}

NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

dataset = list(zip(dataList, obsList))

########
# Start training and testing
########

# remove the saved models to fairly check total training time
try:
    os.remove('saved_models/apply2x2_stable_models.pkl')
except OSError:
    pass

time1 = time.time()
NeurASPobj.learn(dataset, epoch=3, bar=True, task='apply2x2')
print('--- total time for training: %s seconds ---' % int((time.time() - time1)) )
acc, _ = NeurASPobj.testNN('op', testLoader)
print('Test Acc: {:0.2f}%'.format(acc))
print('--- total time from beginning: %s seconds ---' % int((time.time() - startTime)) )