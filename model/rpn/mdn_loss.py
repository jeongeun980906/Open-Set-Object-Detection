import math
from numpy import NaN
import torch
from torch.autograd import Variable
import torch.distributions as TD

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)
def mdn_loss(pred_objectness_logits,mask,data):
    """
    pi: [N x K x HWA]
    mu: [N x K x HWA]
    sigma: [K x HWA]
    mask : [N x HWA]
    data: [D]
    """
    # K = pred_objectness_logits['pi'].size(1)
    # print(mask.shape)
    mask_1,mask_2 = torch.where(mask==True)

    pi = pred_objectness_logits['pi'][mask_1,:,mask_2] # [D x K]
    mu = pred_objectness_logits['mu'][mask_1,:,mask_2] # [D x K]
    mu = torch.sigmoid(mu)
    sigma = pred_objectness_logits['sigma'][mask_1,:,mask_2] # [D x K]
    data_usq = torch.unsqueeze(data,1) # [D x 1]
    data_exp = data_usq.expand_as(sigma) # [D x K]
    probs = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data_exp-mu)/sigma)**2) / sigma # [D x K]
    prob = torch.sum(probs*pi,dim=1) # [D]
    prob = torch.clamp(prob,min=1e-8) # Clamp if the prob is to small
    nll = -torch.log(prob) # [N] 
    out = {'data_usq':data_usq,'data_exp':data_exp,
           'probs':probs,'prob':prob,'nll':nll}
    return nll

def find_logit(pred_objectness_logits):
    '''
    pi: [N x K x D]
    mu: [N x K x D]
    '''
    pi = pred_objectness_logits['pi'][0]
    mu = pred_objectness_logits['mu'][0]
    return [torch.sum(pi*mu,dim=1)]


def mdn_uncertainties(pred_objectness_logits):
    '''
    pi: [N x K x D]
    mu: [N x K x D]
    sigma: [N x K x D]
    '''
    pi = pred_objectness_logits['pi'][0]
    mu = pred_objectness_logits['mu'][0]
    sigma = pred_objectness_logits['sigma'][0]
    # Compute Epistemic Uncertainty
    # M = 0.1# 0.1
    # pi = torch.softmax(M*pi,1) # (optional) heuristics 

    mu_avg = torch.sum(torch.mul(pi,mu),dim=1).unsqueeze(1) # [N x 1 x D]
    mu_exp = mu_avg.expand_as(mu) # [N x K x D]
    mu_diff_sq = torch.square(mu-mu_exp) # [N x K x D]
    epis_unct = torch.sum(torch.mul(pi,mu_diff_sq), dim=1)  # [N x D]

    # Compute Aleatoric Uncertainty
    alea_unct = torch.sum(torch.mul(pi,sigma), dim=1)  # [N x D]
    # Sqrt
    epis_unct,alea_unct = torch.sqrt(epis_unct),torch.sqrt(alea_unct)
    # entropy of pi
    entropy_pi  = -pi*torch.log(pi+1e-8)
    entropy_pi  = torch.sum(entropy_pi,1) #[N x D]
    # print(epis_unct.shape,alea_unct.shape)
    out = {'epis':epis_unct,'alea':alea_unct,'pi_entropy':entropy_pi}
    return out