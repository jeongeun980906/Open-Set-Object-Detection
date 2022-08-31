import torch
import wandb
import argparse
import json

from torch.optim import optimizer
from engine.trainer import SimpleTrainer

from  config.config import get_cfg
from engine.predictor import DefaultPredictor
from model.rcnn import GeneralizedRCNN

from data.build import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from data.mapper import DatasetMapper
import data.transforms as T
from data.phase_1 import load_voc_instances,VOC_CLASS_NAMES
from data.phase_2 import load_openimage_instances,LANDMARK
from structures.image_list import ImageList
from engine.detection_checkpointer import DetectionCheckpointer
from data.utils import build_augmentation
from layers.wrappers import cat
from eval.gmm import GaussianMixture
from eval.maha import maha_distance

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel
# import torch.multiprocessing as mp
# from engine.launch import launch

# python -m torch.distributed.launch --nproc_per_node=4 train_voc.py

def GAUSSIAN(cfg, save_id):
    split = 'train'

    """
    Create configs and perform basic setups.
    """
    RPN_NAME = 'base'
    ROI_NAME = 'mln' if cfg.MODEL.ROI_HEADS.USE_MLN else 'base'
    MODEL_NAME = RPN_NAME + ROI_NAME
    # default_setup(cfg, args)
    # backbone = build_backbone(cfg)
    # head = Res5ROIHeads(cfg,backbone.output_shape())
    rcnn = GeneralizedRCNN(cfg).to('cuda')
    state_dict = torch.load('./ckpt/baseline/{}_{}_15000.pt'.format(save_id,MODEL_NAME),map_location='cuda')
    pretrained_dict = {k: v for k, v in state_dict.items() if k in rcnn.state_dict()}
    rcnn.load_state_dict(pretrained_dict)
    rcnn.eval()

    DIR_NAME = '/data/jeongeun/OWOD_datasets/VOC2007'
    data = load_voc_instances(DIR_NAME,split,VOC_CLASS_NAMES)
    mapper = DatasetMapper(is_train=True, augmentations=build_augmentation(cfg,True))
    data_loader = build_detection_train_loader(data,mapper=mapper,total_batch_size=4)
    record_epis = []
    record_alea = []
    with torch.no_grad():
        for e, batch_in in enumerate(data_loader):
            batch_out = rcnn(batch_in, 1.0)
            for a in batch_out:
                pred_classes = a['instances'].pred_classes
                index = torch.where(pred_classes<20)
                epis = a['instances'].epis[index].cpu().numpy().tolist()
                alea = a['instances'].alea[index].cpu().numpy().tolist()
                record_epis.extend(epis)
                record_alea.extend(alea)
            if e % 100 ==0:
                print(e, len(data)/8)
            if e > 2000:
                break
        # record_epis.append(epis.cpu())

        # alea = [a['instances'].alea for a in batch_out]
        # alea = cat(alea)
        # record_alea.append(alea)

    record_epis = torch.FloatTensor(record_epis)
    record_alea = torch.FloatTensor(record_alea)
    PATH_NAME = './ckpt/{}/{}_{}.json'.format(cfg.MODEL.SAVE_IDX,MODEL_NAME)
    log = {'epis_mean':torch.mean(record_epis).item(), 
                'epis_std': torch.std(record_epis).item(),
                'alea_mean':torch.mean(record_alea).item(), 
                'alea_std': torch.std(record_alea).item()}
    # rcnn.eval()
    # data = load_voc_instances(DIR_NAME,'test',VOC_CLASS_NAMES)
    # mapper = DatasetMapper(is_train=False, augmentations=build_augmentation(cfg,False))
    # data_loader = build_detection_test_loader(data,mapper=mapper)
    with open(PATH_NAME, "w") as json_file:
        json.dump(log, json_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int,default=1,help='save index')
    parser.add_argument('--gpu', type=int,default=0,help='gpu index')
    parser.add_argument('--base_roi', action='store_true', default=False,help='use mln for ROIHEAD')
    parser.add_argument('--log', action='store_true', default=False,help='use wandb')
    args = parser.parse_args()

    print(torch.cuda.device_count())
    torch.cuda.set_device(args.gpu)
    print(torch.cuda.current_device())
    cfg = get_cfg()
    cfg.merge_from_file('./config_files/voc.yaml')
    cfg.phase = 'voc'
    
    cfg.MODEL.SAVE_IDX = args.id
    cfg.MODEL.ROI_HEADS.USE_MLN=1-args.base_roi
    cfg.MODEL.RPN.AUTO_LABEL = False
    cfg.log = args.log

    # cfg.merge_from_list(args.opts)
    RPN_NAME = 'base'
    ROI_NAME = 'mln' if cfg.MODEL.ROI_HEADS.USE_MLN else 'base'
    MODEL_NAME = RPN_NAME + ROI_NAME

    NUM_CLASSES = 21
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

    cfg.freeze()
   
    GAUSSIAN(cfg,args.id)