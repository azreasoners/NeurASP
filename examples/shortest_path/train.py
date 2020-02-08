import sys
sys.path.append("../../")
import time

import torch

from neurasp import NeurASP
from network import FC
from dataGen import obsList, obsListTest, dataList, dataListTest

#############################
# DeepLPMLN program
#############################

nnRule = '''
nn(sp(24, g), [true, false]).
'''

aspRule = '''
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
'''

remove_con = '''
% [nr] 1. No removed edges should be predicted
mistake :- sp(X), removed(X).
'''

path_con = '''
% [p] 2. Prediction must form simple path(s)
% that is: the degree of nodes should be either 0 or 2
mistake :- X=0..15, #count{Y: sp(X,Y)} = 1.
mistake :- X=0..15, #count{Y: sp(X,Y)} >= 3.
'''

reach_con = '''
% [r] 3. Every 2 nodes in the prediction must be reachable
reachable(X, Y) :- sp(X, Y).
reachable(X, Y) :- reachable(X, Z), sp(Z, Y).
mistake :- sp(X, _), sp(Y, _), not reachable(X, Y).
'''

opt_con = '''
% [o] 4. Predicted path should contain least edges
:~ sp(X). [1, X]
'''


########
# Construct nnMapping, set optimizers, and initialize DeepLPMLN object
########

m = FC(40, 50, 50, 50, 50, 50, 24)
nnMapping = {'sp': m}
optimizers = {'sp': torch.optim.Adam(m.parameters(), lr=0.001)}

# training using rules p-r-o-nr
# NeurASPobj = DeepLPMLN(nnRule+aspRule+remove_con+path_con+reach_con+opt_con, nnMapping, optimizers)

# training using rules p-r-o
NeurASPobj = NeurASP(nnRule+aspRule+path_con+reach_con+opt_con, nnMapping, optimizers)

# training using rules p-r
# NeurASPobj = DeepLPMLN(nnRule+aspRule+path_con+reach_con, nnMapping, optimizers)

# training using rules p
# NeurASPobj = DeepLPMLN(nnRule+aspRule+path_con, nnMapping, optimizers)


########
# Start training and testing on a list of different MVPP programs
########
mvppList = [remove_con, path_con, reach_con, remove_con+path_con, remove_con+reach_con, path_con+reach_con, remove_con+path_con+reach_con, remove_con+path_con+reach_con+opt_con]
mvppList = [aspRule+i for i in mvppList]

for idx, constraint in enumerate(mvppList):
    print('Constraint {} is\n{}\n-------------------'.format(idx+1, constraint))

startTime = time.time()
for i in range(50):
    print('Continuously training for 10 epochs round {}...'.format(i+1))
    time1 = time.time()
    NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=10, opt=True, smPickle='data/stableModels.pickle')
    time2 = time.time()
    NeurASPobj.testConstraint(dataList=dataListTest, obsList=obsListTest, mvppList=mvppList)
    print("--- train time: %s seconds ---" % (time2 - time1))
    print("--- test time: %s seconds ---" % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )
