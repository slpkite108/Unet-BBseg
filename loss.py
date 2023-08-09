import torch

def intersection_over_union(pred_coord, target_coord):
    
    x1 = torch.max(pred_coord[:,1], target_coord[:,1])
    y1 = torch.max(pred_coord[:,0], target_coord[:,0])
    x2 = torch.max(pred_coord[:,3], target_coord[:,3])
    y2 = torch.max(pred_coord[:,2], target_coord[:,2])
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    box1_area = abs((pred_coord[:,3] - pred_coord[:,1]) * (pred_coord[:,2] - pred_coord[:,0]))
    box2_area = abs((target_coord[:,3] - target_coord[:,1]) * (target_coord[:,2] - target_coord[:,0]))
    
    iou = intersection / (box1_area + box2_area - intersection + 1e-6)

    return torch.min(iou, torch.ones_like(iou))