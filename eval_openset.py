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
parser.add_argument('--gpu_vit', type=int,default=1,help='gpu index')
parser.add_argument('--af', type=str,default='mul',help='acquisition function',
            choices=["sum", "mul"])
parser.add_argument('--base_rpn', action='store_true', default=False,help='use mdn for RPN')
parser.add_argument('--base_roi', action='store_true', default=False,help='use mln for ROIHEAD')
parser.add_argument('--unct', action='store_true', default=False,help='use mln unct else DINO')
args = parser.parse_args()

print(torch.cuda.device_count())
torch.cuda.set_device(args.gpu)
print(torch.cuda.current_device())
DIR_NAME = '/data/jeongeun/OWOD_datasets/VOC2007' #'/data/opensets/voc/VOCdevkit/VOC2007'
coco_eval = PascalVOCDetectionEvaluator(DIR_NAME,VOC_CLASS_NAMES)
coco_eval.reset()
cfg = get_cfg()
cfg.MODEL.SAVE_IDX = args.id
cfg.MODEL.ROI_HEADS.NUM_CLASSES  = 20 +1
cfg.MODEL.ROI_BOX_HEAD.USE_FD = False
cfg.MODEL.RPN.USE_MDN=1-args.base_rpn
cfg.MODEL.ROI_HEADS.USE_MLN=1-args.base_roi
cfg.MODEL.RPN.AUTO_LABEL_TYPE = args.af
cfg.MODEL.ROI_HEADS.UNCT = args.unct
cfg.MODEL.ROI_HEADS.AF = 'baseline'
cfg.MODEL.RPN.AUTO_LABEL = False

RPN_NAME = 'mdn' if cfg.MODEL.RPN.USE_MDN else 'base'
ROI_NAME = 'mln' if cfg.MODEL.ROI_HEADS.USE_MLN else 'base'
MODEL_NAME = RPN_NAME + ROI_NAME

device = 'cuda'
model = GeneralizedRCNN(cfg,device).to(device)
state_dict =  torch.load('./ckpt/{}/{}_{}_15000.pt'.format(cfg.MODEL.ROI_HEADS.AF,cfg.MODEL.SAVE_IDX,MODEL_NAME),map_location=device)
state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
model.load_state_dict(state_dict)
model.eval()
if not args.unct:
    model.load_ssl(args.gpu_vit)

data = load_voc_instances(DIR_NAME,'test',VOC_CLASS_NAMES,phase=None,COCO_CLASS=True)#[::100]
print(len(data))
mapper = DatasetMapper(is_train=False, augmentations=build_augmentation(cfg,False))
data_loader = build_detection_test_loader(data,mapper=mapper,batch_size=4)

for e, instance in enumerate(data_loader):
    with torch.no_grad():
        output = model(instance,1.0)
        coco_eval.process(instance,output)
    # break
    if e%100==0:
        print(e)
        # break
eval_res = coco_eval.evaluate()
print(eval_res)
