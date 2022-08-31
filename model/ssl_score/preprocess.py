import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from structures.image_list import ImageList
from structures.instances import Instances
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from typing import List
from layers.wrappers import cat
import json

PIXEL_MEAN = torch.tensor([103.530, 116.280, 123.675]).view(-1,1,1)
PIXEL_STD = torch.tensor([1.0, 1.0, 1.0]).view(-1,1,1)
TF = tf.Compose([
                tf.Resize((256,256)),
                tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

TF_CLIP = tf.Compose([
                tf.Resize(224,interpolation=tf.InterpolationMode.BICUBIC),
                tf.CenterCrop(224),
                tf.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

def preprocess(image:torch.tensor,boxes:torch.tensor,gt_boxes:torch.tensor,CLIP=False):
    '''
    erase gt boxes with 0
    Crop image + Pad + Resize to (256,256)
    Args:
        image: [3 x H x W]
        boxes: proposals [N x 4]
        gt_boxes: [o x 4]
    Returns:
        cropped image [N x 256 x 256]
    '''
    res = []
    boxes = boxes.type(torch.LongTensor).detach()
    # print(image.shape)
    w = image.shape[2]
    h = image.shape[1]
    boxes[:,0].clamp(min=0,max=h)
    boxes[:,1].clamp(min=0,max=w)
    boxes[:,2].clamp(min=0,max=h)
    boxes[:,3].clamp(min=0,max=w)
    for e,box in enumerate(boxes):
        # print(image.shape)
        # print(box)
        # box = [100,100,400,400]
        if abs(box[1]-box[3]) == 0:
            box[3]+=1
        if abs(box[0]-box[2])==0:
            box[2]+=1
        # print(image.shape,box)
        crop_image = image[:,box[1]:box[3],box[0]:box[2]].cpu()
        crop_image = (crop_image*PIXEL_STD)+PIXEL_MEAN
        width,height = crop_image.shape[1], crop_image.shape[2]
        length = max(width,height)
        pad = (int((length-height+1)/2),int((length-height)/2),
                int((length-width+1)/2),int((length-width)/2))

        # demo = crop_image.cpu().numpy().astype(np.uint8).transpose(1,2,0)
        # plt.figure()
        # plt.imshow(demo)
        # plt.savefig('./dummy/proposals/crop.png')
        # print(crop_image.max())
        if CLIP:
            crop_image = F.pad(input=crop_image,pad = pad)/255
            crop_image = TF_CLIP(crop_image)
        else:
            crop_image = F.pad(input=crop_image,pad = pad)/255
            crop_image = TF(crop_image)
        res.append(crop_image.unsqueeze(0))
    # print(res)
    res = cat(res)
    return res


def open_candidate(path:str = './log.json'):
    with open(path,'r') as jf:
        data = json.load(jf)
    pos_data = data['fg']
    neg_data = data['bg']
    return [torch.FloatTensor(pos_data),torch.FloatTensor(neg_data)]