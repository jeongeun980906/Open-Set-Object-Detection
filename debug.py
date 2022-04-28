from engine.predictor import DefaultPredictor
from config.config import get_cfg
from model.rcnn import GeneralizedRCNN
import torch
import cv2
import matplotlib.pyplot as plt
# from utils.visualizer import Visualizer
from data.catalog import MetadataCatalog, DatasetCatalog

print(torch.cuda.device_count())
torch.cuda.set_device(3)
print(torch.cuda.current_device())

cfg = get_cfg()
cfg.MODEL.ROI_HEADS.NUM_CLASSES  = 20
cfg.MODEL.RPN.USE_MDN = True
model = GeneralizedRCNN(cfg).to('cuda')
state_dict = torch.load('./ckpt/mdn_17500.pt')
model.load_state_dict(state_dict)
predictor = DefaultPredictor(cfg,model)


img = cv2.imread('input.png')
predictor.uncertainty_predict(img)