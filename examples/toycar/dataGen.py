import numpy as np
import torch
import torchvision
# from skimage.transform import resize
from torch.autograd import Variable
from yolo.utils.utils import load_classes, non_max_suppression
from yolo.utils.datasets import ImageFolder
from yolo.models import Darknet

def termPath2dataList(termPath, img_size):
    """
    @param termPath: a string of the form 'term path' denoting the path to the files represented by term
    """
    factsList = []
    dataList = []
    term, path = termPath.split(' ')
    dataloader = torch.utils.data.DataLoader(
        ImageFolder(path, img_size=img_size), 
        batch_size=1, 
        shuffle=False
    )
    for _, img in dataloader:
        img = Variable(img.type(torch.FloatTensor))
        with torch.no_grad():
            output = yolo(img)
            facts, dataDic = postProcessing(output, term)
            factsList.append(facts)
            dataList.append(dataDic)        
    return factsList, dataList

def postProcessing(output, term, num_classes=80, conf_thres=0.3, nms_thres=0.4):
    facts = ''
    dataDic = {}
    detections = non_max_suppression(output, num_classes, conf_thres, nms_thres)
    if detections:
        for detection in detections:
            for idx, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detection):
                terms = '{},b{}'.format(term, idx)
                facts += 'box({}, {}, {}, {}, {}).\n'.format(terms, int(x1), int(y1), int(x2), int(y2))
                className = '{}'.format(cls_name[int(cls_pred)])
                X = torch.zeros([1, 4], dtype=torch.float64)
                if className in domain:
                    X[0, domain.index(className)] += round(float(cls_conf), 3)
                else:
                    X[0, -1] += round(float(cls_conf), 3)
                dataDic[terms] = X
    return facts, dataDic

########
# Load Yolo network, which is used to generate the facts for bounding boxes
########

config_path = './yolo/yolov3.cfg'
weights_path = './yolo/yolov3.weights'
cls_name = load_classes('./yolo/coco.names')
img_size = 416
yolo = Darknet(config_path, img_size=416)
yolo.load_weights(weights_path)
yolo.eval()

########
# Construct a list of facts and a list of dataDic, where each dataDic maps terms to tensors
########

# set the term and the path to the image files represetned by this term
termPath = 'img ./data/'
# set up the set of classes that we consider
domain = ["person", "car", "truck", "other"]
factsList, dataList = termPath2dataList(termPath, img_size)