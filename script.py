import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image

import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import torchvision.models.quantization as models
import matplotlib.pyplot as plt
import time
import os
import sys
import shutil
import cv2
import dlib

def test(model):
    model.train(False)
    
    output = model(image)
    _, predicted = torch.max(output, 1)
        
    return int(1 - predicted)

def create_combined_model(model_fe):
    model_fe_features = nn.Sequential(
        model_fe.quant,
        model_fe.conv1,
        model_fe.maxpool,
        model_fe.stage2,
        model_fe.stage3,
        model_fe.stage4,
        model_fe.conv5,
        model_fe.dequant,
    )

    new_head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(65536, 2),
    )

    new_model = nn.Sequential(
        model_fe_features,
        nn.Flatten(1),
        new_head,
    )
    return new_model
	
use_gpu = torch.cuda.is_available()

model_fe = models.shufflenet_v2_x1_0(pretrained=True, progress=True, quantize=True)

new_model = create_combined_model(model_fe)

if use_gpu:
    new_model = new_model.cuda()
    new_model.load_state_dict(torch.load('ShuffleNet.pth'))
else:
	new_model.load_state_dict(torch.load('ShuffleNet.pth', map_location=torch.device('cpu')))

detector = dlib.get_frontal_face_detector()
result = dict()
data_dir = sys.argv[1]
for photo in os.listdir(data_dir):
    image = cv2.imread(os.path.join(data_dir, photo))
    face_rects = list(detector(image, 1))
    if len(face_rects) == 0:
        result[os.path.join(data_dir, photo)] = -1
    else:
        rect = face_rects[0]
#         cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 0), 2)
        height = rect.bottom() - rect.top()
        width = rect.right() - rect.left()
        image = image[max(0, int(rect.top() - height / 2)):int(rect.bottom() + height / 2),
                      max(0, int(rect.left() - width / 2)):int(rect.right() + width / 2)]
        image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image).unsqueeze(0)
        if use_gpu:
            image = image.cuda()
        result[os.path.join(data_dir, photo)] = test(new_model)

with open('result.csv', 'w') as f:
    for key in result.keys():
        f.write("%s,%s\n"%(key,result[key]))