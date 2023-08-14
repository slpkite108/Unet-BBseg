import numpy as np
from PIL import Image
import os

def draw_filter(labels,bboxes_coord, type, n_organ, img_size = (512,512)):     
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

def save_image(np_image, path, name):
    img = Image.fromarray(np_image,mode='L')
    img.save(os.path.join(path,name+'.png'),'png')

