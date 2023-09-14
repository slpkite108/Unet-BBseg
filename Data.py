import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from utils import draw_filter
import torchvision.transforms.functional as FT

class FPUS(torch.utils.data.Dataset):
    def __init__(self, image_data, target_data, transform=None):
        self.image_data = image_data
        self.target_data = target_data
        self.transform = transform

        self.label_to_index = {'abdomen':1, 'arm':2, 'legs':3, 'head':4}
        self.num_labels = 4
        self.padding = ((141,142),(0,0))#udlr
        self.resize_scale = 1/672
    def __len__(self):
        return len(self.image_data)
        
    def __getitem__(self, idx):
        image_path = self.image_data[idx]
        image = np.array(Image.open(image_path).convert("L"))
        image = np.pad(image,self.padding)
        image = cv2.resize(image,dsize=(600,600))
        image = FT.to_tensor(image)
        
        target = self.target_data[idx]

        labels = torch.LongTensor([self.label_to_index[label] for label in target['label']])
        
        boxes = torch.FloatTensor(list(zip(target['xtl'],target['ytl'],target['xbr'],target['ybr'])))
        boxes[:, [1, 3]] += self.padding[0][0]

        boxes *= self.resize_scale
        
        if self.transform:
            image,boxes,labels = self.transform(image,boxes,labels)
        
        return image,boxes,labels
    
    def collate_fn(self, batch):

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each