import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import os


distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']

label_name = ['abdomen', 'arm_A','arm_B', 'legs_A','legs_B', 'head']

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

def bb_normalize(bb_coord, shape):#shape : (H,W)
    return torch.hstack(bb_coord[0]/shape[0],bb_coord[1]/shape[1],bb_coord[2]/shape[0],bb_coord[3]/shape[0])


def xy_align(xy):
    # xy 텐서의 shape: (batch_size, class, coord)
    
    min_x = torch.min(xy[:, :, 0], xy[:, :, 2])  # 최소 x 좌표
    min_y = torch.min(xy[:, :, 1], xy[:, :, 3])  # 최소 y 좌표
    max_x = torch.max(xy[:, :, 0], xy[:, :, 2])  # 최대 x 좌표
    max_y = torch.max(xy[:, :, 1], xy[:, :, 3])  # 최대 y 좌표
    
    aligned_xy = torch.stack([min_x, min_y, max_x, max_y], dim=-1)
    # aligned_xy 텐서의 shape: (batch_size, class, 4)
    
    return aligned_xy

def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max

def visualize(images,overlaps,isTruth=False):
    #image.shape = (H,W)
    #overlap.shape = (N,4)
    annotated_images = []
    images = images.to('cpu')
    overlaps = xy_align(overlaps.to('cpu'))

    font = ImageFont.load_default()
    #font = font.font_variant(size=15)
    for image, overlap in zip(images, overlaps):
        
        annotated_image = Image.fromarray(image.numpy().squeeze()).convert('RGB')
        draw = ImageDraw.Draw(annotated_image)

        
        for i in range(overlap.shape[0]):
            if len(np.unique(overlap[i])) > 1:
                box_location = overlap[i,[1,0,3,2]].tolist()
                draw.rectangle(xy=box_location, outline=distinct_colors[i])
                draw.rectangle(xy=[l+1. for l in box_location], outline = distinct_colors[i])

                text_size = font.getbbox(label_name[i].upper())
                #print(text_size)
                text_location = [box_location[0]+2., box_location[1]-text_size[3]]
                textbox_location = [box_location[0], box_location[1]-text_size[3], box_location[0]+text_size[2]+4., box_location[1]]
                draw.rectangle(xy = textbox_location, fill=distinct_colors[i])
                draw.text(xy=text_location, text=label_name[i].upper(), fill='white' if isTruth else 'black', font=font)
        del draw
        annotated_images.append(list(np.array(annotated_image).transpose((2,0,1))))

    annotated_images = torch.tensor(np.array(annotated_images))
    return annotated_images

def save_checkpoint(save_path,model,epoch,optimizer):
    if not os.path.exists(os.path.join(save_path,'pt')):
        os.mkdir(os.path.join(save_path,'pt'))

    state = {
        'model':model,
        'epoch':epoch,
        'optimizer':optimizer
    }

    torch.save(state, os.path.join(save_path,'pt',f'model_epoch_{epoch}.pt'))
    print(f"Model saved at epoch {epoch}")