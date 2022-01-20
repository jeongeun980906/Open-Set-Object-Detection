from eval.voc_eval import PascalVOCDetectionEvaluator
from config.config import get_cfg
from model.rcnn import GeneralizedRCNN
import torch

from data.build import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from data.utils import build_augmentation
from data.mapper import DatasetMapper
import data.transforms as T
from data.phase_1 import load_voc_instances,VOC_CLASS_NAMES
import os

print(torch.cuda.device_count())
torch.cuda.set_device(3)
print(torch.cuda.current_device())
DIR_NAME = '/data/private/OWOD/datasets/VOC2007' #'/data/opensets/voc/VOCdevkit/VOC2007'
coco_eval = PascalVOCDetectionEvaluator(DIR_NAME,VOC_CLASS_NAMES)
coco_eval.reset()
cfg = get_cfg()
cfg.MODEL.ROI_HEADS.NUM_CLASSES  = 20
cfg.MODEL.RPN.USE_MDN = True
model = GeneralizedRCNN(cfg).to('cuda')
state_dict = torch.load('./ckpt/mdn_17500.pt')
model.load_state_dict(state_dict)

model.eval()
data = load_voc_instances(DIR_NAME,'known_test',VOC_CLASS_NAMES)
mapper = DatasetMapper(is_train=False, augmentations=build_augmentation(cfg,False))
data_loader = build_detection_test_loader(data,mapper=mapper,batch_size=8)

for e, instance in enumerate(data_loader):
    with torch.no_grad():
        output = model(instance)
        coco_eval.process(instance,output)
    # break
    if e%100==0:
        print(e)
eval_res = coco_eval.evaluate()
print(eval_res)
