import sys
sys.path.append('../../')
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

dprogram = '''
nn(mrp(24, g), [true, false]).
'''

aspProgram = '''
mrp(X) :- mrp(X,g,true).

mrp(0,1) :- mrp(0).
mrp(1,2) :- mrp(1).
mrp(2,3) :- mrp(2).
mrp(4,5) :- mrp(3).
mrp(5,6) :- mrp(4).
mrp(6,7) :- mrp(5).
mrp(8,9) :- mrp(6).
mrp(9,10) :- mrp(7).
mrp(10,11) :- mrp(8).
mrp(12,13) :- mrp(9).
mrp(13,14) :- mrp(10).
mrp(14,15) :- mrp(11).
mrp(0,4) :- mrp(12).
mrp(4,8) :- mrp(13).
mrp(8,12) :- mrp(14).
mrp(1,5) :- mrp(15).
mrp(5,9) :- mrp(16).
mrp(9,13) :- mrp(17).
mrp(2,6) :- mrp(18).
mrp(6,10) :- mrp(19).
mrp(10,14) :- mrp(20).
mrp(3,7) :- mrp(21).
mrp(7,11) :- mrp(22).
mrp(11,15) :- mrp(23).

mrp(X,Y) :- mrp(Y,X).

:- X=0..15, #count{Y: mrp(X,Y)} = 1.
:- X=0..15, #count{Y: mrp(X,Y)} >= 3.
reachable(X, Y) :- mrp(X, Y).
reachable(X, Y) :- reachable(X, Z), mrp(Z, Y).
:- mrp(X, _), mrp(Y, _), not reachable(X, Y).
'''

########
# Define nnMapping and optimizers, initialze NeurASP object
########

m = FC(40, *[50, 50, 50, 50, 50], 24)
nnMapping = {'mrp': m}
optimizer = {'mrp': torch.optim.Adam(m.parameters(), lr=0.001)}
NeurASPobj = NeurASP(dprogram+aspProgram, nnMapping, optimizer)

########
# Start training and testing
########

saveModelPath = 'data/model.pt'

for i in range(20):
    print('Continuously training for 10 epochs round {}...'.format(i+1))
    time1 = time.time()
    NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=10, opt=True, smPickle='data/stableModels.pickle', bar=True)
    time2 = time.time()
    NeurASPobj.testConstraint(dataList=dataListTest, obsList=obsListTest, mvppList=[aspProgram])
    print("--- train time: %s seconds ---" % (time2 - time1))
    print("--- test time: %s seconds ---" % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )

print('Storing the trained model into {}'.format(saveModelPath))
torch.save(m.state_dict(), saveModelPath)
