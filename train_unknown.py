import torch
import wandb
import argparse

from torch.optim import optimizer
from engine.trainer import SimpleTrainer

from  config.config import get_cfg
from model.rcnn import GeneralizedRCNN
from engine.optimizer import build_optimizer,build_lr_scheduler

from data.build import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from data.mapper import DatasetMapper
import data.transforms as T
from data.phase_1 import load_voc_instances,VOC_CLASS_NAMES,COCO_CLASS_NAMES
from data.phase_2 import load_openimage_instances,LANDMARK
from structures.image_list import ImageList
from engine.detection_checkpointer import DetectionCheckpointer
from data.utils import build_augmentation

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel
# import torch.multiprocessing as mp
# from engine.launch import launch

# python -m torch.distributed.launch --nproc_per_node=4 train_voc.py

def main(cfg, phase,args):
    DIR_NAME = '/data/jeongeun/OWOD_datasets/VOC2007'
    split = 'train'
    if phase == None:
        CLASS_NAME = COCO_CLASS_NAMES
        use_coco = True
    else:
        CLASS_NAME = VOC_CLASS_NAMES
        use_coco = False
    """
    Create configs and perform basic setups.
    """
    # default_setup(cfg, args)
    # backbone = build_backbone(cfg)
    # head = Res5ROIHeads(cfg,backbone.output_shape())
    rcnn = GeneralizedRCNN(cfg).to('cuda')
    optim = build_optimizer(cfg,rcnn)
    schedular = build_lr_scheduler(cfg,optim)
    checkpointer = DetectionCheckpointer(
                # Assume you want to save checkpoints together with logs/statistics
                rcnn,
                cfg.OUTPUT_DIR
            )
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
    rcnn = checkpointer.model
    if cfg.MODEL.RPN.AUTO_LABEL:
        rcnn.proposal_generator.MODEL.to(args.gpu_vit)
    if phase == 'l':
        data = load_openimage_instances(DIR_NAME,split)
    else:
        data = load_voc_instances(DIR_NAME,split,CLASS_NAME,phase,use_coco)
    mapper = DatasetMapper(is_train=True, augmentations=build_augmentation(cfg,True))
    data_loader = build_detection_train_loader(data,mapper=mapper,total_batch_size=8)

    trainer = SimpleTrainer(rcnn,data_loader,optim,cfg.log)
    RPN_NAME = 'base'
    ROI_NAME = 'mln' if cfg.MODEL.ROI_HEADS.USE_MLN else 'base'
    MODEL_NAME = RPN_NAME + ROI_NAME
    print('data size: ',len(data))
    for i in range(cfg.SOLVER.MAX_ITER):
        # print(i)
        trainer.run_step(i/cfg.SOLVER.MAX_ITER)
        schedular.step()
        if i%500 == 0:
            print(i)
        if i%5000==0:
            print(i)
            torch.save(trainer.model.state_dict(),'./ckpt/baseline/{}_{}_{}.pt'.format(cfg.MODEL.SAVE_IDX,MODEL_NAME,i))
    torch.save(trainer.model.state_dict(),'./ckpt/baseline/{}_{}_{}.pt'.format(cfg.MODEL.SAVE_IDX,MODEL_NAME,i))
    # rcnn.eval()
    # data = load_voc_instances(DIR_NAME,'test',VOC_CLASS_NAMES)
    # mapper = DatasetMapper(is_train=False, augmentations=build_augmentation(cfg,False))
    # data_loader = build_detection_test_loader(data,mapper=mapper)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int,default=1,help='save index')
    parser.add_argument('--gpu', type=int,default=0,help='gpu index')
    parser.add_argument('--gpu_vit', type=int,default=1,help='gpu index')
    parser.add_argument('--af', type=str,default='base',help='acquisition function',
                choices=["base", "sum"])
    parser.add_argument('--auto_label', action='store_true', default=False,help='Auto Labeling')

    parser.add_argument('--all', action='store_true', default=False,help='All COCO training')

    parser.add_argument('--base_roi', action='store_true', default=False,help='use mln for ROIHEAD')
    parser.add_argument('--CLIP', action='store_true', default=False,help='use CLIP')
    parser.add_argument('--log', action='store_true', default=False,help='use wandb')
    args = parser.parse_args()
    
    if args.all:
        phase = None
    else:
        phase =  't1'
    print(torch.cuda.device_count())
    torch.cuda.set_device(args.gpu)
    print(torch.cuda.current_device())
    cfg = get_cfg()
    cfg.merge_from_file('./config_files/voc.yaml')
    cfg.phase = 'voc'

    cfg.MODEL.SAVE_IDX = args.id
    cfg.MODEL.ROI_HEADS.USE_MLN=1-args.base_roi
    cfg.MODEL.RPN.AUTO_LABEL = args.auto_label
    cfg.log = args.log
    cfg.gpu_vit = "cuda:{}".format(args.gpu_vit)
    cfg.MODEL.RPN.AUTO_LABEL_TYPE = args.af

    if phase == None:
        NUM_CLASSES = 80
    else: 
        NUM_CLASSES = 20
    if args.auto_label:
        NUM_CLASSES += 1
    print(cfg.MODEL.ROI_HEADS.USE_MLN)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    # cfg.merge_from_list(args.opts)
    RPN_NAME = 'base'
    ROI_NAME = 'mln' if cfg.MODEL.ROI_HEADS.USE_MLN else 'base'
    MODEL_NAME = RPN_NAME + ROI_NAME
    cfg.freeze()
    if args.log:
        wandb.init(config=cfg,tags= str(args.id),name = '{}_{}_{}'.format(args.id, MODEL_NAME),project='faster rcnn')
    main(cfg,phase,args)