import torch
import json
import math

PATH_NAME = './ckpt/uncertainty/1_mdn.json'
THRES = 0.05
LOG_THRES = -1e9#math.log(0.01)

def filter_unct(pred_classes,unct,path=None):
    with open(PATH_NAME,'r') as jf:
        data = json.load(jf)
    mean = data['epis_mean'] + data['alea_mean'] 
    std = math.sqrt(data['epis_std']**2 + data['alea_std']**2)
    dist = torch.distributions.normal.Normal(torch.tensor([mean]).to('cuda'),torch.tensor([std]).to('cuda'))
    cdf = 1-dist.cdf(unct)
    change_indx = torch.where(cdf<THRES)[0]
    pred_classes[change_indx] = 20
    # print(change_indx)
    return pred_classes


def filter_fd(pred_classes, log_prob):
    print(log_prob)
    change_indx = torch.where(log_prob<LOG_THRES)[0]
    pred_classes[change_indx] = 20
    print(change_indx)
    return pred_classes