from config.config import get_cfg
from model.rcnn import GeneralizedRCNN
import torch
import argparse

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
from tools.memory import retry_if_cuda_oom
from model.sampling import subsample_labels

from model.ssl_score.preprocess import preprocess,open_candidate
from model.ssl_score.score import cosine_distance_torch
import random

cfg = get_cfg()
cfg.merge_from_file('./config_files/voc.yaml')
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
cfg.MODEL.ROI_HEADS.USE_MLN = False
torch.cuda.set_device(0)
print(torch.cuda.current_device())


DIR_NAME = '/data/private/OWOD/datasets/VOC2007'
data = load_voc_instances(DIR_NAME,'test',VOC_CLASS_NAMES,phase=None,COCO_CLASS=True)
data = data[::50]
print(len(data))
mapper = DatasetMapper(is_train=True, augmentations=build_augmentation(cfg,False))
data_loader = build_detection_test_loader(data,mapper=mapper,batch_size=1)

device_vit = 'cuda:1'
referenec_set = open_candidate().to('cuda')
vit = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device_vit)

EPOCHS = [5000]#[2000*(i+1) for i in range(8)]
for e in EPOCHS:
    model = GeneralizedRCNN(cfg).to('cuda')
    state_dict = torch.load('./ckpt/baseline/2_basebase_{}.pt'.format(e))
    model.load_state_dict(state_dict)
    model.eval()
    total_score = {
        'known': [],
        'unknown': [],
        'background': [],
        'ignore': []
    }

    obj_score = {
        'known': [],
        'unknown': [],
        'background': [],
        'ignore': []
    }

    dino_score = {
        'known': [],
        'unknown': [],
        'background': [],
        'ignore': []
    }


    def clip(tensor,box_size):
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].
        Args:
            box_size (height, width): The clipping box's size.
        """
        h, w = box_size
        x1 = tensor[:, 0].clamp(min=0, max=w)
        y1 = tensor[:, 1].clamp(min=0, max=h)
        x2 = tensor[:, 2].clamp(min=0, max=w)
        y2 = tensor[:, 3].clamp(min=0, max=h)
        return torch.stack((x1, y1, x2, y2), dim=-1)

    for batched_inputs in data_loader:
        images = model.preprocess_image(batched_inputs)
        images = images.tensor
        w = images.shape[2]
        h = images.shape[3]
        features = model.backbone(images)

        features = [features['res4']]
        anchors = model.proposal_generator.anchor_generator(features)
        pred_objectness_logits, pred_anchor_deltas = model.proposal_generator.head(features)

        # Transform
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, model.proposal_generator.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
                ]

        pred_proposals = model.proposal_generator._decode_proposals(
                    anchors, pred_anchor_deltas)
        
        pred_proposals = pred_proposals[0]
        pred_objectness_logits = pred_objectness_logits[0]

        for i,(proposal_box,proposal_score) in enumerate(zip(pred_proposals,pred_objectness_logits)):        
            
            valid_mask = torch.isfinite(proposal_box).all(dim=1)
            keep = batched_nms(proposal_box,proposal_score,valid_mask,0.3)
            post_proposal_box = proposal_box[keep]
            post_proposal_score = torch.sigmoid(proposal_score[keep]).detach()
            gt_instance = batched_inputs[i]['instances']
            gt_boxes = gt_instance.gt_boxes
            gt_labels = gt_instance.gt_classes
            IoU = retry_if_cuda_oom(pairwise_iou)(gt_boxes.to('cuda'), Boxes(post_proposal_box))
            matched_idxs, matched_labels = retry_if_cuda_oom(model.proposal_generator.anchor_matcher)(IoU)
            label = gt_labels[matched_idxs]
            pos_idx, neg_idx = subsample_labels(
                        matched_labels, 10, 0.8, 0
                    )

            # Some samples from ignore
            ignore = nonzero_tuple(matched_labels == -1)[0]
            perm = torch.randperm(ignore.numel(),device=ignore.device)[:3]
            ignore_idx = ignore[perm]

            # print(neg_idx)
            post_proposal_box = clip(post_proposal_box,[h,w]).type(torch.LongTensor)
            pos_gt_label = label[pos_idx]

            unk_mask = (pos_gt_label==20)
            unk_idx = pos_idx[unk_mask]

            pos_mask = (pos_gt_label<20)
            pos_idx = pos_idx[pos_mask]
            positive_boxes = post_proposal_box[pos_idx]
            positive_scores = post_proposal_score[pos_idx]

            unk_boxes = post_proposal_box[unk_idx]
            unk_scores = post_proposal_score[unk_idx]

            neg_boxes = post_proposal_box[neg_idx]
            neg_scores = post_proposal_score[neg_idx]

            ign_boxes = post_proposal_box[ignore_idx]
            ign_scores = post_proposal_score[ignore_idx]
            
            ## Scoring
            if pos_idx.shape[0] != 0:
                positive_patch = preprocess(images[i],positive_boxes,None)
                feat = vit(positive_patch.to(device_vit)).detach().to('cuda')
                pos_cos_sim = cosine_distance_torch(referenec_set,feat)
                pos_total_score = positive_scores.to('cuda')*pos_cos_sim

                dino_score['known'] += pos_cos_sim.cpu().numpy().tolist()
                obj_score['known'] += positive_scores.cpu().numpy().tolist()
                total_score['known'] += pos_total_score.cpu().numpy().tolist()
                print(torch.mean(pos_total_score))
            if unk_idx.shape[0] != 0:
                unk_patch = preprocess(images[i],unk_boxes,None)
                feat = vit(unk_patch.to(device_vit)).detach().to('cuda')
                unk_cos_sim = cosine_distance_torch(referenec_set,feat)
                unk_total_score =unk_scores.to('cuda')*unk_cos_sim

                dino_score['unknown'] += unk_cos_sim.cpu().numpy().tolist()
                obj_score['unknown'] += unk_scores.cpu().numpy().tolist()
                total_score['unknown'] += unk_total_score.cpu().numpy().tolist()
                print(torch.mean(unk_total_score))

            if neg_idx.shape[0] != 0:
                neg_patch = preprocess(images[i],neg_boxes,None)
                feat = vit(neg_patch.to(device_vit)).detach().to('cuda')
                neg_cos_sim = cosine_distance_torch(referenec_set,feat)
                neg_total_score =neg_scores.to('cuda')*neg_cos_sim

                dino_score['background'] += neg_cos_sim.cpu().numpy().tolist()
                obj_score['background'] += neg_scores.cpu().numpy().tolist()
                total_score['background'] += neg_total_score.cpu().numpy().tolist()
                print(torch.mean(neg_total_score))
            
            if ignore_idx.shape[0] != 0:
                ign_patch = preprocess(images[i],ign_boxes,None)
                feat = vit(ign_patch.to(device_vit)).detach().to('cuda')
                ign_cos_sim = cosine_distance_torch(referenec_set,feat)
                ign_total_score =ign_scores.to('cuda')*ign_cos_sim

                dino_score['ignore'] += ign_cos_sim.cpu().numpy().tolist()
                obj_score['ignore'] += ign_scores.cpu().numpy().tolist()
                total_score['ignore'] += ign_total_score.cpu().numpy().tolist()
                print(torch.mean(ign_total_score))

    NUM_SAMPLES = 500
    plt.figure()
    for key in dino_score:
        samples = random.choices(dino_score[key],k=NUM_SAMPLES)
        counts, bins = np.histogram(samples,bins=40)
        if key == 'ignore':
            plt.hist(bins[:-1], bins, weights=counts, alpha=0.2,label = key)
        else:
            plt.hist(bins[:-1], bins, weights=counts, alpha=0.5,label = key)
    plt.ylim((0,100))
    plt.legend()
    plt.savefig('./dummy/hist/dino_score_{}.png'.format(e))

    plt.figure()
    for key in obj_score:
        samples = random.choices(obj_score[key],k=NUM_SAMPLES)
        counts, bins = np.histogram(samples,bins=40)
        if key == 'ignore':
            plt.hist(bins[:-1], bins, weights=counts, alpha=0.2,label = key)
        else:
            plt.hist(bins[:-1], bins, weights=counts, alpha=0.5,label = key)
    plt.ylim((0,100))
    plt.legend()
    plt.savefig('./dummy/hist/objectness_{}.png'.format(e))

    plt.figure()
    for key in total_score:
        samples = random.choices(total_score[key],k=NUM_SAMPLES)
        counts, bins = np.histogram(samples,bins=40)
        if key == 'ignore':
            plt.hist(bins[:-1], bins, weights=counts, alpha=0.2,label = key)
        else:
            plt.hist(bins[:-1], bins, weights=counts, alpha=0.5,label = key)
    plt.ylim((0,100))
    plt.legend()
    plt.savefig('./dummy/hist/total_score_{}.png'.format(e))