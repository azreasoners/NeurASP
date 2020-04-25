import sys
sys.path.append('../../')

import torch

from dataGen import dataList, postProcessing
from neurasp import NeurASP
from yolo.models import Darknet

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram = r'''
image(img).
nn(label(yolo,X), ["person", "car", "truck", "other"]) :- image(X).
'''

aspProgram = r'''
% define smaller/2 between labels
smaller("cup", "cat").
smaller("cat", "person").
smaller("person", "car").
smaller("person", "truck").
smaller(X,Y) :- smaller(X,Z), smaller(Z,Y).

% define smaller/3 between objects in bounding boxes
smaller(I,B1,B2) :- not -smaller(I,B1,B2), label(I,B1,L1), label(I,B2,L2), smaller(L1,L2).
-smaller(I,B2,B1) :- box(I,B1,X1,Y1,X2,Y2), box(I,B2,X1',Y1',X2',Y2'), Y2>=Y2', |X1-X2|*|Y1-Y2| < |X1'-X2'|*|Y1'-Y2'|.
smaller(I,B1,B2) :- -smaller(I,B2,B1).
toy(I,B1) :- label(I,B1,L1), label(I,B2,L2), smaller(I,B1,B2), smaller(L2,L1).
'''

########
# Define nnMapping and optimizers, initialze NeurASP object
########

config_path = './yolo/yolov3.cfg'
weights_path = './yolo/yolov3.weights'
img_size = 416
m = Darknet(config_path, img_size=img_size)
m.load_weights(weights_path)
nnMapping = {'label': m}

NeurASPobj = NeurASP(dprogram, nnMapping, optimizers=None)

########
# Start inference
########

for idx, dataDic in enumerate(dataList):
    models = NeurASPobj.infer(dataDic=dataDic, obs='', mvpp=aspProgram, postProcessing=postProcessing)
    print('\nInfernece Result on Data {}:'.format(idx+1))
    print(models[0])
