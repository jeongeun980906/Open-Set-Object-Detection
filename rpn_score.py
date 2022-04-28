import torch
import argparse

from model.ssl_score.obj import simple_cnn
from config.config import get_cfg
from model.rcnn import GeneralizedRCNN

from data.build import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from data.utils import build_augmentation
from data.mapper import DatasetMapper
import data.transforms as T
from data.phase_1 import load_voc_instances,VOC_CLASS_NAMES
import os
from layers.nms import batched_nms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
from layers.wrappers import nonzero_tuple

from structures.box import Boxes,pairwise_iou
from layers.wrappers import cat
from tools_det.memory import retry_if_cuda_oom
from model.sampling import subsample_labels

from model.ssl_score.preprocess import preprocess,open_candidate
from model.ssl_score.dino_score import cosine_distance_torch,save_images
import random
# from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance 

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int,default=1,help='save index')
parser.add_argument('--gpu', type=int,default=0,help='gpu index')
parser.add_argument('--gpu_vit', type=int,default=1,help='gpu index')
parser.add_argument('--af', type=str,default='mul',help='acquisition function',
            choices=["sum", "mul"])
parser.add_argument('--base_rpn', action='store_true', default=False,help='use mdn for RPN')
parser.add_argument('--CLIP', action='store_true', default=False,help='use mdn for RPN')
parser.add_argument('--base_roi', action='store_true', default=False,help='use mln for ROIHEAD')
args = parser.parse_args()

print(torch.cuda.device_count())
torch.cuda.set_device(args.gpu)
print(torch.cuda.current_device())

DIR_NAME = '/data/jeongeun/OWOD_datasets/VOC2007' #'/data/opensets/voc/VOCdevkit/VOC2007'
cfg = get_cfg()
cfg.MODEL.SAVE_IDX = args.id
cfg.MODEL.ROI_HEADS.NUM_CLASSES  = 20 +1
cfg.MODEL.ROI_BOX_HEAD.USE_FD = False
cfg.MODEL.RPN.USE_MDN=1-args.base_rpn
cfg.MODEL.ROI_HEADS.USE_MLN=1-args.base_roi
cfg.MODEL.RPN.AUTO_LABEL_TYPE = args.af
cfg.MODEL.RPN.USE_CLIP = args.CLIP

RPN_NAME = 'mdn' if cfg.MODEL.RPN.USE_MDN else 'base'
ROI_NAME = 'mln' if cfg.MODEL.ROI_HEADS.USE_MLN else 'base'
MODEL_NAME = RPN_NAME + ROI_NAME

torch.cuda.set_device(1)
print(torch.cuda.current_device())

data = load_voc_instances(DIR_NAME,'test',VOC_CLASS_NAMES,phase=None,COCO_CLASS=True)
# data = data[::100]
# print(len(data))
mapper = DatasetMapper(is_train=True, augmentations=build_augmentation(cfg,False))
data_loader = build_detection_test_loader(data,mapper=mapper,batch_size=4)

# device_vit = 'cuda:2'
# referenec_set = open_candidate()[0].to('cuda')
# vit = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device_vit)

model = GeneralizedRCNN(cfg).to('cuda')
state_dict =  torch.load('./ckpt/{}/{}_{}_15000.pt'.format(cfg.MODEL.ROI_HEADS.AF,cfg.MODEL.SAVE_IDX,MODEL_NAME))
state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
model.load_state_dict(state_dict)
model.eval()

pos_obj,unk_obj = [] , []
unk_obj_c, pos_obj_c = 0, 0
num_pos, num_unk = 0,0 
with torch.no_grad():
    for batched_inputs in data_loader:
        images = model.preprocess_image(batched_inputs)
        images = images.tensor
        w = images.shape[2]
        h = images.shape[3]
        features = model.backbone(images)

        features = [features['res4']]
        gt_instances = [x['instances'].to('cuda') for x in batched_inputs]
        anchors = model.proposal_generator.anchor_generator(features)
        pred_objectness_logits, pred_anchor_deltas = model.proposal_generator.head(features)

        # Transform
        if args.base_rpn:
            pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            torch.sigmoid(score.permute(0, 2, 3, 1).flatten(1))
            for score in pred_objectness_logits
            ]
        else:
            pred_objectness_logits = [
            # (N, K, A, Hi, Wi) -> (N, K, Hi, Wi, A) -> (N, K, Hi*Wi*A)
            torch.sum(
            score['pi'].permute(0, 1, 3, 4, 2).flatten(2)*
            torch.sigmoid(score['mu'].permute(0, 1, 3, 4, 2).flatten(2)),
            dim=1)
                for score in pred_objectness_logits]

        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, model.proposal_generator.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
                ]

        pred_proposals = model.proposal_generator._decode_proposals(
                    anchors, pred_anchor_deltas)
        
        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        gt_labels = [x.gt_classes for x in gt_instances]
        anchors = Boxes.cat(anchors)

        i = 0
        for image_size_i, gt_boxes_i,gt_labels_i in zip(image_sizes, gt_boxes,gt_labels):
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            if args.base_rpn:
                matched_idxs, matched_labels = retry_if_cuda_oom(model.proposal_generator.anchor_matcher)(match_quality_matrix) # [anchor h/16 w/16]
            else:
                matched_idxs, matched_labels,matched_iou = retry_if_cuda_oom(model.proposal_generator.anchor_matcher)(match_quality_matrix) # [anchor h/16 w/16]
            del match_quality_matrix
            obj_idx = (matched_labels ==  1)
            label = gt_labels_i[matched_idxs]

            obj = pred_objectness_logits[0][i]
            obj = obj[obj_idx]
            label = label[obj_idx]
            # obj = torch.sigmoid(obj[obj_idx])

            pos_idx = (label<20)
            unk_idx = (label==20)

            pos_obj_tensor = obj[pos_idx]
            unk_obj_tensor = obj[unk_idx]

            i+=1
            num_pos += pos_obj_tensor.size(0)
            num_unk += unk_obj_tensor.size(0)
            thrs = 0.5 if args.base_rpn else 0.5
            
            pos_obj += pos_obj_tensor.cpu().numpy().tolist()
            unk_obj += unk_obj_tensor.cpu().numpy().tolist()

            pos_obj_c += (pos_obj_tensor>thrs).sum()#
            unk_obj_c += (unk_obj_tensor>thrs).sum() #

print('pos accuracy: %.3f'%(pos_obj_c/num_pos))
print('unk accuracy: %.3f'%(unk_obj_c/num_unk))
# objectness = pos_obj + unk_obj
# print(sum(unk_obj)/len(unk_obj))
# GT = [1]*len(pos_obj) + [0]*len(unk_obj)

# auroc = roc_auc_score(GT,objectness)
# print(auroc)

pos_obj = np.asarray(pos_obj)
histogram1 = np.histogram(pos_obj.ravel(), bins=100, range=[0, 1])
unk_obj = np.asarray(unk_obj)
histogram2 = np.histogram(unk_obj.ravel(), bins=100, range=[0, 1])
w = wasserstein_distance(histogram1[0],histogram2[0])
print(w)