import torch
from torch.autograd import Variable
from yolo.utils.utils import load_classes, non_max_suppression
from yolo.utils.datasets import ImageFolder
from yolo.models import Darknet

def termPath2dataList(termPath, img_size, domain):
    """
    @param termPath: a string of the form 'term path' denoting the path to the files represented by term
    """
    factsList = []
    dataList = []
    # Load Yolo network, which is used to generate the facts for bounding boxes and a tensor for each bounding box
    config_path = './yolo/yolov3.cfg'
    weights_path = './yolo/yolov3.weights'
    yolo = Darknet(config_path, img_size)
    yolo.load_weights(weights_path)
    yolo.eval()

    # feed each image into yolo
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
            facts, dataDic = postProcessing(output, term, domain)
            factsList.append(facts)
            dataList.append(dataDic)        
    return factsList, dataList

def postProcessing(output, term, domain, num_classes=80, conf_thres=0.3, nms_thres=0.4):
    facts = ''
    dataDic = {}
    cls_name = load_classes('./yolo/coco.names')
    detections = non_max_suppression(output, num_classes, conf_thres, nms_thres)
    if detections:
        for detection in detections:
            for idx, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detection):
                terms = '{},b{}'.format(term, idx)
                facts += 'box({}, {}, {}, {}, {}).\n'.format(terms, int(x1), int(y1), int(x2), int(y2))
                className = '{}'.format(cls_name[int(cls_pred)])
                X = torch.zeros([1, len(domain)], dtype=torch.float64)
                if className in domain:
                    X[0, domain.index(className)] += round(float(cls_conf), 3)
                else:
                    X[0, -1] += round(float(cls_conf), 3)
                dataDic[terms] = X
    return facts, dataDic
