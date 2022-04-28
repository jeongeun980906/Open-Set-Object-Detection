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

PIXEL_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1,1,1)
PIXEL_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1,1,1)


def autolabel_clip(images:ImageList,proposals:List[Instances],
                gt_instances:List[Instances],anchor_matcher,
                model: torch.nn.Module,text:torch.Tensor):
    res = []
    thres = 0.8
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
        if score.shape[0]>100:
            top_score, top_index = torch.topk(score,k=100)
            top_boxes = boxes[top_index,:]
        else:
            top_score = score
            top_boxes = boxes
        # print(top_boxes.shape)
        patches = preprocess(image,top_boxes,gt_boxes,CLIP=True)
        device = top_score.device
        device_vit = next(model.parameters()).device
        ssl_score = 1 - compute_score(patches,model,text,device,device_vit)

        keep = nms(top_boxes,ssl_score,0.1)
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
        # print(auto_label.shape)
        res.append(auto_label)
    return res


def compute_score(patches,model,text,device,device_vit):
    logits_per_image, logits_per_text = model(patches.to(device_vit), text.to(device_vit))
    probs = (logits_per_image/2).softmax(dim=-1).to(device)
    bg_probs = probs[:,0] + probs[:,1] + probs[:,2]
    return bg_probs


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