import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from utils import draw_filter

class FPUS(torch.utils.data.Dataset):
    def __init__(self, image_data, target_data, transform=None):
        self.image_data = image_data
        self.target_data = target_data
        self.transform = transform

        self.label_to_index = {'abdomen':0, 'armA':1,'armB':2, 'legsA':3,'legsB':4, 'head':5}
        self.num_labels = 6
        self.padding = ((141,142),(0,0))#udlr
        self.resize_scale = 600/672
    def __len__(self):
        return len(self.image_data)
        
    def __getitem__(self, idx):
        image_path = self.image_data[idx]
        image = np.array(Image.open(image_path).convert("L"))
        image = np.pad(image,self.padding)
        image = cv2.resize(image,dsize=(600,600))
        
        target = self.target_data[idx]
        temp_index = {'abdomen':0, 'arm':1,'legs':3,'head':5}
        labels_index = np.array([temp_index[label] for label in target['label']])
        
        indices_of_ones = np.where(labels_index == 1)[0]
        if len(indices_of_ones) >= 2:
            labels_index[indices_of_ones[1]] = 2
            
        indices_of_threes = np.where(labels_index == 3)[0]
        if len(indices_of_threes) >= 2:
            labels_index[indices_of_threes[1]] = 2
            
        labels = np.zeros(self.num_labels)
        labels[labels_index]=1
        
        bboxes_coord = np.array(list(zip(target['ytl'],target['xtl'],target['ybr'],target['xbr'])))
        bboxes_coord[:, [0, 2]] += self.padding[0][0]
        bboxes_coord *= self.resize_scale
        
        for i in range(self.num_labels):
            if labels[i]<1:
                bboxes_coord = np.insert(bboxes_coord,i,[0.,0.,0.,0.],axis=0)
        
        bbox = draw_filter(labels_index,np.array(bboxes_coord), 'bbox', self.num_labels, image.shape)
        
        #bbox_v = Variate_Bbox(img.shape, np.array(bboxes_coord),1000, 6)
        #bbox_img = np.concatenate((image.reshape((1,image.shape[0],image.shape[1])), bbox), axis=0)
        
        bbox_conv = bbox*image
        
        data_dict = {
            'image':image.reshape((1,image.shape[0],image.shape[1])),
            'labels':labels,
            #'labelIndex':labels_index,
            'bbcoord':bboxes_coord,
            'bbox':bbox,
            #'bbox_img':bbox_img,
            #'seed':bbox_v,
            'bbconv':bbox_conv,
            'index':idx
        }
        
        if self.transform:
            data_dict = self.transform(data_dict)
        
        return data_dict