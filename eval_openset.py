from eval.openset_eval import PascalVOCDetectionEvaluator
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

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int,default=1,help='save index')
parser.add_argument('--gpu', type=int,default=0,help='gpu index')
parser.add_argument('--af', type=str,default='baseline',help='acquisition function',
            choices=["baseline", "uncertainty"])
parser.add_argument('--base_rpn', action='store_true', default=False,help='use mdn for RPN')
parser.add_argument('--base_roi', action='store_true', default=False,help='use mln for ROIHEAD')
args = parser.parse_args()

print(torch.cuda.device_count())
torch.cuda.set_device(args.gpu)
print(torch.cuda.current_device())
DIR_NAME = '/data/private/OWOD/datasets/VOC2007' #'/data/opensets/voc/VOCdevkit/VOC2007'
coco_eval = PascalVOCDetectionEvaluator(DIR_NAME,VOC_CLASS_NAMES)
coco_eval.reset()
cfg = get_cfg()
cfg.MODEL.SAVE_IDX = args.id
cfg.MODEL.ROI_HEADS.NUM_CLASSES  = 21
cfg.MODEL.ROI_BOX_HEAD.USE_FD = True
cfg.MODEL.ROI_HEADS.AUTO_LABEL = True
cfg.MODEL.RPN.USE_MDN=1-args.base_rpn
cfg.MODEL.ROI_HEADS.USE_MLN=1-args.base_roi
cfg.MODEL.ROI_HEADS.AF = args.af

MODEL_NAME = 'mdn' if cfg.MODEL.RPN.USE_MDN else 'base'
model = GeneralizedRCNN(cfg).to('cuda')
state_dict = torch.load('./ckpt/{}/{}_17500.pt'.format(cfg.MODEL.ROI_HEADS.AF,MODEL_NAME))
model.load_state_dict(state_dict)
if args.base_roi:
    model.roi_heads.box_predictor.load_gmm()

model.eval()
data = load_voc_instances(DIR_NAME,'test',VOC_CLASS_NAMES,phase=None,COCO_CLASS=True)
print(len(data))
mapper = DatasetMapper(is_train=False, augmentations=build_augmentation(cfg,False))
data_loader = build_detection_test_loader(data,mapper=mapper,batch_size=8)

for e, instance in enumerate(data_loader):
    with torch.no_grad():
        output = model(instance)
        coco_eval.process(instance,output)
    # break
    if e%100==0:
        print(e)
        # break
eval_res = coco_eval.evaluate()
print(eval_res)
