from save_unkowns import MAX_ITER
import torch
import cv2
import copy
import matplotlib.pyplot as plt
from engine.predictor import DefaultPredictor
import numpy as np
import random
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


MAX_ITER = 20
ROOT = './dummy/roi_head/'
SEED = 100
torch.manual_seed(seed=SEED)

np.random.seed(seed=SEED)
random.seed(SEED)
print(torch.cuda.device_count())
torch.cuda.set_device(2)
print(torch.cuda.current_device())
cfg = get_cfg()
cfg.merge_from_file('./config_files/voc.yaml')
cfg.MODEL.RPN.USE_MDN=True
cfg.MODEL.ROI_HEADS.USE_MLN=True
cfg.log = False 
cfg.MODEL.ROI_HEADS.AUTO_LABEL = True
cfg.MODEL.ROI_HEADS.AF = 'uncertainty'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21
cfg.INPUT.RANDOM_FLIP = "none"
# cfg.merge_from_list(args.opts)
cfg.freeze()
# wandb.init(config=cfg,tags= 'temp',name = 'temp',project='temp')

DIR_NAME = '/data/jeongeun/OWOD_datasets/VOC2007'
split = 'train'
model = GeneralizedRCNN(cfg).to('cuda')
state_dict = torch.load('./ckpt/uncertainty/1_mdn_17000.pt')
model.load_state_dict(state_dict)
predictor = DefaultPredictor(cfg,model)
data = load_voc_instances(DIR_NAME,split,VOC_CLASS_NAMES,phase=None,eval_OS=True)
print(len(data))
mapper = DatasetMapper(is_train=True, augmentations=build_augmentation(cfg,True))
data_loader = build_detection_train_loader(data,mapper=mapper,total_batch_size=1)

IDXS = random.sample(range(5000, 10000), MAX_ITER)
VOC_CLASS_NAMES_NEW = (*VOC_CLASS_NAMES, 'unknown')

RPN_NAME = 'mdn' if cfg.MODEL.RPN.USE_MDN else 'base'
ROI_NAME = 'mln' if cfg.MODEL.ROI_HEADS.USE_MLN else 'base'
MODEL_NAME = RPN_NAME + ROI_NAME

for i,IDX in enumerate(IDXS):
    batched_inputs = data.__getitem__(IDX)
    gts = batched_inputs['annotations']
    gt_bboxs = [a['bbox'] for a in gts]
    gt_cls = [a['category_id'] for a in gts]
    file_name = batched_inputs['file_name']
    img = cv2.imread(file_name)

    pred = predictor(img)
    pred = pred['instances']._fields
    pred_boxes = pred['pred_boxes']
    scores = pred['scores']
    pred_classes = pred['pred_classes']
    print(pred_classes)
    index = torch.where(scores>0.2)[0]

    top_pred_boxes = pred_boxes[index]
    top_pred_classes = pred_classes[index]

    demo_image = copy.deepcopy(img)
    for bbox,label in zip(top_pred_boxes,top_pred_classes):
        if label==20:
            color = (0,255,255)
        else:
            color = (255,0,0)
        cv2.rectangle(demo_image, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.putText(demo_image, VOC_CLASS_NAMES_NEW[int(label)], 
                                (int(bbox[0]), int(bbox[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(ROOT+'{}_{}_{}.png'.format(cfg.MODEL.ROI_HEADS.AF,MODEL_NAME,i),demo_image)