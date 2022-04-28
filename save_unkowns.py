import torch
import weakref
import cv2
import copy
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.optim import optimizer
from engine.trainer import SimpleTrainer
import argparse

from  config.config import get_cfg
from model.rcnn import GeneralizedRCNN
from engine.optimizer import build_optimizer,build_lr_scheduler

from data.build import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from data.mapper import DatasetMapper
import data.transforms as T
from data.phase_1 import load_voc_instances,VOC_CLASS_NAMES
from structures.image_list import ImageList
from engine.detection_checkpointer import DetectionCheckpointer
from data.utils import build_augmentation


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int,default=1,help='save index')
parser.add_argument('--gpu', type=int,default=0,help='gpu index')
parser.add_argument('--af', type=str,default='baseline',help='acquisition function',
            choices=["baseline", "uncertainty"])
parser.add_argument('--base_rpn', action='store_true', default=False,help='use mdn for RPN')
parser.add_argument('--base_roi', action='store_true', default=False,help='use mln for ROIHEAD')
parser.add_argument('--log', action='store_true', default=False,help='use wandb')
args = parser.parse_args()

MAX_ITER = 20
ROOT = './dummy/proposals/'
SEED = 1000
torch.manual_seed(seed=SEED)

np.random.seed(seed=SEED)
random.seed(SEED)

print(torch.cuda.device_count())
torch.cuda.set_device(3)
print(torch.cuda.current_device())
cfg = get_cfg()
cfg.merge_from_file('./config_files/voc.yaml')
cfg.MODEL.SAVE_IDX = args.id
cfg.MODEL.RPN.USE_MDN=1-args.base_rpn
cfg.MODEL.ROI_HEADS.USE_MLN=1-args.base_roi
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21
cfg.MODEL.ROI_HEADS.AUTO_LABEL = True
cfg.log = args.log
cfg.MODEL.ROI_HEADS.AF = args.af
# cfg.merge_from_list(args.opts)
RPN_NAME = 'mdn' if cfg.MODEL.RPN.USE_MDN else 'base'
ROI_NAME = 'mln' if cfg.MODEL.ROI_HEADS.USE_MLN else 'base'
MODEL_NAME = RPN_NAME + ROI_NAME

# cfg.merge_from_list(args.opts)
cfg.freeze()
# wandb.init(config=cfg,tags= 'temp',name = 'temp',project='temp')

DIR_NAME = '/data/jeongeun/OWOD_datasets/VOC2007'
split = 'train'
rcnn = GeneralizedRCNN(cfg).to('cuda')
state_dict = torch.load('./ckpt/{}/{}_{}_17500.pt'.format(cfg.MODEL.ROI_HEADS.AF,args.id,MODEL_NAME))
rcnn.load_state_dict(state_dict)
data = load_voc_instances(DIR_NAME,split,VOC_CLASS_NAMES)
mapper = DatasetMapper(is_train=True, augmentations=build_augmentation(cfg,True))
data_loader = build_detection_train_loader(data,mapper=mapper,total_batch_size=1)
loader = iter(data_loader)

for i, batched_inputs in enumerate(data_loader):
# for i in range(MAX_ITER):
#     batched_inputs = next(loader)
    if i>MAX_ITER:
        break
    images = rcnn.preprocess_image(batched_inputs)
    features = rcnn.backbone(images.tensor)
    gt_instances = [x['instances'].to(rcnn.device) for x in batched_inputs]
    proposals, proposal_losses = rcnn.proposal_generator(images, features, gt_instances)
    # images, features, proposals, gt_instances
    proposals = rcnn.roi_heads.label_and_sample_proposals(proposals, gt_instances)

    image = batched_inputs[0]['image']
    inputs = {"image": batched_inputs[0]['image'], 
                "height": batched_inputs[0]['height'], "width": batched_inputs[0]['width']}
    proposals = rcnn._postprocess(proposals,[inputs],images.image_sizes)

    gt_classes = proposals[0]['instances']._fields['gt_classes']
    # gt_classes = proposals[0]._fields['gt_classes']
    unkown_index = torch.where(gt_classes==20)
    known = torch.where(gt_classes<20)

    proposal_boxes = proposals[0]['instances']._fields['proposal_boxes'].tensor
    # proposal_boxes = proposals[0]._fields['proposal_boxes'].tensor
    proposal_boxes_unkown = proposal_boxes[unkown_index]
    proposal_boxes_known = proposal_boxes[known]

    file_name = batched_inputs[0]['file_name']
    demo_image = cv2.imread(file_name)

    for bbox in proposal_boxes_known:
        # print(bbox)
        bbox = bbox.tolist()
        # cv2.rectangle(demo_image, (0,0), (100,100), (0, 255, 0), 2)
        cv2.rectangle(demo_image, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]),int(bbox[3])), (0, 255, 0), 1)

    for bbox in proposal_boxes_unkown:
        # print(bbox)
        bbox = bbox.tolist()
        # cv2.rectangle(demo_image, (0,0), (100,100), (0, 255, 0), 2)
        cv2.rectangle(demo_image, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]),int(bbox[3])), (255, 255, 0), 2)
    cv2.imwrite(ROOT+'{}_{}_{}.png'.format(cfg.MODEL.ROI_HEADS.AF,MODEL_NAME,i),demo_image)