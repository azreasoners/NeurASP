import sys
sys.path.append('../../')
import time

import torch
from torch.autograd import Variable

from dataGen import GridProbData
from network import FC
from neurasp import NeurASP

dprogram = '''
nn(sp(24, g), [true, false]).

sp(X) :- sp(g,X,true).

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

:- X=0..15, #count{Y: sp(X,Y)} = 1.
:- X=0..15, #count{Y: sp(X,Y)} >= 3.
reachable(X, Y) :- sp(X, Y).
reachable(X, Y) :- reachable(X, Z), sp(Z, Y).
:- sp(X, _), sp(Y, _), not reachable(X, Y).
'''

dprogram_test = '''
sp(X) :- sp(g,X,true).

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

:- X=0..15, #count{Y: sp(X,Y)} = 1.
:- X=0..15, #count{Y: sp(X,Y)} >= 3.
reachable(X, Y) :- sp(X, Y).
reachable(X, Y) :- reachable(X, Z), sp(Z, Y).
:- sp(X, _), sp(Y, _), not reachable(X, Y).
'''

m = FC(40, *[50, 50, 50, 50, 50], 24)

nnMapping = {'sp': m}

optimizer = {'sp':torch.optim.Adam(m.parameters(), lr=0.001)}

NeurASPobj = NeurASP(dprogram, nnMapping, optimizer)


# process the data 
dataset = GridProbData('data/data.txt')

dataList = []
obsList = []

for i, d in enumerate(dataset.train_data):
    d_tensor = Variable(torch.from_numpy(d).float(), requires_grad=False)
    dataList.append({'g': d_tensor})

with open('data/evidence_train.txt', 'r') as f:
    obsList = f.read().strip().strip('#evidence').split('#evidence')

# testing 
dataListTest = []
obsListTest = []

for d in dataset.test_data:
    d_tensor = Variable(torch.from_numpy(d).float(), requires_grad=False)
    dataListTest.append({'g': d_tensor})

with open('data/evidence_test.txt', 'r') as f:
    obsListTest = f.read().strip().strip('#evidence').split('#evidence')

########
# Start training and testing
########
startTime = time.time()
for i in range(20):
    print('Continuously training for 10 epochs round {}...'.format(i+1))
    time1 = time.time()
    NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=10, opt=True, smPickle='data/stableModels.pickle')
    time2 = time.time()
    NeurASPobj.testConstraint(dataList=dataListTest, obsList=obsListTest, mvppList=[dprogram_test])
    print("--- train time: %s seconds ---" % (time2 - time1))
    print("--- test time: %s seconds ---" % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )
