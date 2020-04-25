import numpy as np
import torch
import torchvision
# from skimage.transform import resize
from torch.autograd import Variable
from yolo.utils.utils import load_classes, non_max_suppression
from yolo.utils.datasets import ImageFolder

img_size = 416
cls_name = load_classes('./yolo/coco.names')
termPath = 'img ./data/'

def postProcessing(output, num_classes=80, conf_thres=0.3, nms_thres=0.4):
    info = []
    detections = non_max_suppression(output, num_classes, conf_thres, nms_thres)
    if detections:
        for detection in detections:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                info.append(("\"{}\"".format(cls_name[int(cls_pred)]), int(x1), int(y1), int(x2), int(y2), round(float(cls_conf), 3)))
    return info

def termPath2dataList(termPath):
    """
    @param termPath: a string of the form 'term path' denoting the path to the files represented by term
    """
    dataList = []
    term, path = termPath.split(' ')
    dataloader = torch.utils.data.DataLoader(
        ImageFolder(path, img_size=img_size), 
        batch_size=1, 
        shuffle=False
    )
    for imgPath, img in dataloader:
        img = Variable(img.type(torch.FloatTensor))
        dataList.append({term: img})
    return dataList

dataList = termPath2dataList(termPath)
