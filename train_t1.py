import torch
import wandb
import weakref
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
from data.phase_1 import load_voc_instances,VOC_CLASS_NAMES
from structures.image_list import ImageList
from engine.detection_checkpointer import DetectionCheckpointer
from data.utils import build_augmentation

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from engine.launch import launch

# python -m torch.distributed.launch --nproc_per_node=4 train_voc.py

def main():
    DIR_NAME = '/data/private/OWOD/datasets/VOC2007'
    split = 'train'

    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file('./config_files/voc.yaml')
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
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

    data = load_voc_instances(DIR_NAME,split,VOC_CLASS_NAMES)
    mapper = DatasetMapper(is_train=True, augmentations=build_augmentation(cfg,True))
    data_loader = build_detection_train_loader(data,mapper=mapper,total_batch_size=8)

    trainer = SimpleTrainer(rcnn,data_loader,optim)

    for i in range(cfg.SOLVER.MAX_ITER):
        # print(i)
        trainer.run_step()
        schedular.step()
        if i%500==0:
            print(i)
            torch.save(trainer.model.state_dict(),'./ckpt/t1_{}.pt'.format(i))

    rcnn.eval()
    data = load_voc_instances(DIR_NAME,'test',VOC_CLASS_NAMES)
    mapper = DatasetMapper(is_train=False, augmentations=build_augmentation(cfg,False))
    data_loader = build_detection_test_loader(data,mapper=mapper)


if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES=3
    cfg = get_cfg()
    cfg.merge_from_file('./config_files/voc.yaml')
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    wandb.init(config=cfg,tags= 'temp',name = 'temp',project='t1')
    main()