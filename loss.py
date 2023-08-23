import torch
#intersection over union
def intersection_over_union(pred_coord, target_coord): #(ytl,xtl,yrb,xrb)
    
    x1 = torch.max(pred_coord[:,:,1], target_coord[:,:,1])
    y1 = torch.max(pred_coord[:,:,0], target_coord[:,:,0])
    x2 = torch.min(pred_coord[:,:,3], target_coord[:,:,3])
    y2 = torch.min(pred_coord[:,:,2], target_coord[:,:,2])
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    #intersection[intersection <= 0.]=0.

    box1_area = abs((pred_coord[:,:,3] - pred_coord[:,:,1]) * (pred_coord[:,:,2] - pred_coord[:,:,0]))
    box2_area = abs((target_coord[:,:,3] - target_coord[:,:,1]) * (target_coord[:,:,2] - target_coord[:,:,0]))
    
    iou = intersection / (box1_area + box2_area - intersection + 1e-6)
    iou[box2_area == 0] = 1-(box1_area[box2_area==0])/(600*600 +1e-6)#둘 다 0이면 1출력

    #print("target: ",target_coord)
    #print("iou: ",iou)
    #print("width: ",x2-x1)
    #print("height",y2-y1)
    #print("intersection: ",intersection)
    #print("box1_area:", box1_area)
    #print("box2_area", box2_area)
    #print("iou area: ",box1_area + box2_area - intersection)

    return iou

#BoundingBoxRegression

#Confidence Score
#Non-Maximum Suppression

#평가지표
#Average Precision
#meanAverge Precision
