import torch
import torchvision.transforms as tf
from typing import List
from structures.image_list import ImageList
from structures.instances import Instances

from .preprocess import preprocess
from structures.box import Boxes,pairwise_ioa,pairwise_iou
from layers.nms import nms
from layers.wrappers import cat
from tools_det.memory import retry_if_cuda_oom

import numpy as np
import matplotlib.pyplot as plt

PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1)
PIXEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1)

def autolabel_dino(images:ImageList,proposals:List[Instances],gt_instances:List[Instances],
                anchor_matcher,model: torch.nn.Module, candidate:List[torch.tensor]
                ,score_type = 'mul'): 
    res = []
    thres = 0.5 if score_type == 'mul' else 1.6 # 0.6 1.5 # 0.4, 1.3
    for image,proposal,gt in zip(images,proposals,gt_instances):
        boxes = proposal.proposal_boxes
        gt_boxes = gt.gt_boxes
        # gt_clasees = gt.gt_classes[0].item()
        IoU = retry_if_cuda_oom(pairwise_iou)(gt_boxes, boxes)
        _, gt_label1 = retry_if_cuda_oom(anchor_matcher)(IoU) # [2000] 
        IoA = retry_if_cuda_oom(pairwise_ioa)(gt_boxes, boxes)
        _, gt_label2 = retry_if_cuda_oom(anchor_matcher)(IoA) # [2000] 
        index_candidate = (gt_label1 == 0) & (gt_label2 == 0)# [2000]
        if index_candidate.sum()==0:
            # print('none')
            return None
        boxes = boxes.tensor[index_candidate,:]
        score = proposal.objectness_logits[index_candidate]
        gt_boxes = gt_boxes.tensor
        if score.shape[0]>10:
            top_score, top_index = torch.topk(score,k=10)
            top_boxes = boxes[top_index,:]
        else:
            top_score = score
            top_boxes = boxes
        # print(top_boxes.shape)
        patches = preprocess(image,top_boxes,gt_boxes)
        device = top_score.device
        device_vit = next(model.parameters()).device
        ssl_score = compute_socre(top_score,patches,model,candidate,score_type,device,device_vit)
        
        keep = nms(top_boxes,ssl_score,0.3)
        ssl_score = ssl_score[keep]
        top_boxes = top_boxes[keep]
        top_score = top_score[keep]
        # print(torch.mean(ssl_score))
        if ssl_score.shape[0]>5:
            k=5
        else:
            k = ssl_score.shape[0]
        s, auto_label_index = torch.topk(ssl_score,k=k)
        # print(s)
        filter = (s>thres)#(s>-0.0) & (s<0.2)
        auto_label_index = auto_label_index[filter]
        # Save Sampled Patches
        # if auto_label_index.shape[0] != 0:
        #     save_images(patches,auto_label_index,0)
        auto_label = top_boxes[auto_label_index,:]
        # print(s,auto_label)
        res.append(auto_label)
    return res
    
def save_images(patches,auto_label_index,save_index):
    if auto_label_index==None:
        sampled_patches= patches
    else:
        sampled_patches = patches[auto_label_index] # [N x 3 x 256 x 256]
    for e,sp in enumerate(sampled_patches):
        sp = ((sp*PIXEL_STD)+PIXEL_MEAN)*255
        sp = sp.cpu().numpy().astype(np.uint8).transpose(1,2,0)
        plt.figure()
        plt.imshow(sp)
        plt.savefig('./dummy/proposals/crop_{}_{}.png'.format(e,save_index))

def compute_socre(score:torch.tensor,
                patches:torch.tensor,
                model:torch.nn.Module,
                candidate_feat: List[torch.tensor],
                score_type:str, # 'mul' or sum
                device: str, device_vit:str = 'cuda:3'):
    # print(patches.device,next(model.parameters()).device)
    feat_total = []
    batch_size = 10
    iter = patches.shape[0]//batch_size
    if iter == 0:
        batch = patches
        feat = model(batch.to(device_vit)) # [N x 768]
        feat_total = feat.cpu().detach().numpy().tolist()
    else:
        for it in range(iter):
            if it == iter-1:
                batch = patches[it*batch_size:,:,:,:]
            else:
                batch = patches[it*batch_size:(it+1)*batch_size,:,:,:]
            feat = model(batch.to(device_vit)) # [N x 768]
            feat = feat.cpu().detach().numpy().tolist()
            feat_total += feat
    ref_feat =  torch.FloatTensor(feat_total)
    # print(torch.mean(ref_feat))
    device_d = ref_feat.device
    # cros_cos_sim = cosine_distance_torch(ref_feat,ref_feat,type='min').to(device)
    cos_sim_pos = cosine_distance_torch(candidate_feat[0].to(device_d),ref_feat).to(device)
    # cos_sim_neg = cosine_distance_torch(candidate_feat[1].to(device_d),ref_feat).to(device)
    if score_type == 'sum':
        return torch.sigmoid(score) + (1-cos_sim_pos)# cos_sim_pos - cos_sim_neg  #torch.sigmoid(score) +
    elif score_type == 'mul':
        return torch.sigmoid(score) * (1-cos_sim_pos)

def cosine_distance_torch(x1, x2=None, eps=1e-8,type='max'):
    # print(x1.shape,x2.shape)
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    distance = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    if type == 'max':
        return torch.max(distance,axis=0).values
    elif type == 'min':
        return torch.min(distance,axis=0).values
    elif type == 'mean':
        return torch.mean(distance,dim=0)