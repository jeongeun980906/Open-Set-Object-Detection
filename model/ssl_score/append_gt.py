import torch
from typing import List
from structures.instances import Instances
from structures.box import Boxes

def append_gt(labels:List[torch.tensor]
        ,GT_instances:List[Instances],
        num_classes :int= 20):
    if labels == None:
        return GT_instances
    res = []
    for label,instance in zip(labels,GT_instances):
        if label.shape[0] == 0:
            res.append(instance)
        else:
            img_size = instance._image_size
            temp = instance.gt_boxes.tensor
            mask = check_size(label)
            label = label[mask,:]
            gt_box = torch.cat((temp,label),dim=0)
            new_gt_box= Boxes(gt_box)
            num = label.shape[0]
            temp = instance.gt_classes
            device = temp.device
            new_gt_classes = torch.cat((temp,torch.LongTensor([num_classes]*num).to(device)),dim=0)
            new_instance = Instances(img_size,
                        gt_boxes=new_gt_box,gt_classes=new_gt_classes)
            res.append(new_instance)
    # print(res)
    return res

def check_size(gt_boxes):
    ws = abs(gt_boxes[:,0]-gt_boxes[:,2]).unsqueeze(-1) # [N]
    hs = abs(gt_boxes[:,1]-gt_boxes[:,3]).unsqueeze(-1) # [N]
    temp = torch.cat((ws,hs),dim=-1) # [N x 2]
    min_length = torch.min(temp,dim=-1).values
    max_length = torch.max(temp,dim=-1).values
    mask = (min_length>50) & (max_length<500)
    return mask