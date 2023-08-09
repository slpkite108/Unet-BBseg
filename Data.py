import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2

def draw_filter(labels,bboxes_coord, typ, n_organ, img_size = (512,512)):     
    noisy_mask = rect_mask(img_size, labels, bboxes_coord, n_organ)
    
    return noisy_mask

def rect_mask(shape, labels, bboxes, n_organs):
    """Given a bbox and a shape, creates a mask (white rectangle foreground, black background)
    Param:
        shape: shape (H,W) or (H,W,1)
        bbox: bbox numpy array [y1, x1, y2, x2]
    Returns:
        mask
    """
    mask = np.zeros((n_organs,shape[0],shape[1]),np.uint8)
    bboxes = bboxes
    if n_organs > 1:
        if len(bboxes[0]) != 1:
            for idx, label in enumerate(labels):
                bbox = bboxes[idx]
                if len(np.unique(bbox)) != 1:
                    mask[label][np.int32(bbox[0]):np.int32(bbox[2]), np.int32(bbox[1]):np.int32(bbox[3])] = 255 
        x = np.array(mask)
    else:
        mask = np.zeros(shape, np.uint8)
        mask[np.int32(bboxes[0]):np.int32(bboxes[2]),
                    np.int(bboxes[1]):np.int32(bboxes[3])] = 255
        x = mask
        
    return x

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
        
"""def FPUS_collate_fn(batch):
    # 데이터 배치의 요소들을 개별적으로 추출하여 딕셔너리의 리스트를 만듦
    data = [item for item in batch]
    print(item['bbox_img'] for item in data)
    # 딕셔너리들을 통해 원하는 전처리 등을 수행
    # 예시로 'input', 'bbox_img', 'name', 'bbox', 'seed', 'bbconv', 'index' 키에 해당하는 값들을 추출
    inputs = np.stack([item['input'] for item in data])
    bbox_img = np.stack([item['bbox_img'] for item in data])
    names = torch.stack([item['name'] for item in data])
    bboxes = np.stack([item['bbox'] for item in data])
    seeds = np.stack([item['seed'] for item in data])
    bbconvs = np.stack([item['bbconv'] for item in data])
    indices = [item['index'] for item in data]
    
    # 데이터를 원하는 형태로 가공한 후 딕셔너리 형태로 반환
    processed_data = {
        'input': inputs,
        'bbox_img': bbox_img,
        'name': names,
        'bbox': bboxes,
        'seed': seeds,
        'bbconv': bbconvs,
        'index': indices
    }

    return processed_data
"""