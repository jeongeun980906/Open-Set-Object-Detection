import numpy as np
import torch
import torch.nn as nn

class MixtureHead_CNN(nn.Module):
    def __init__(self,in_channels,num_anchors,k=5):
        super().__init__()
        self.num_anchors = num_anchors
        self.k = k
        self.sig_max = 2
        self.sig_min = 0.1
        self.cnn_pi = nn.Conv2d(in_channels,num_anchors*k,kernel_size=1,stride=1)
        self.cnn_mu = nn.Conv2d(in_channels,num_anchors*k,stride = 1, kernel_size=1)
        self.cnn_sigma = nn.Conv2d(in_channels,num_anchors*k,stride=1,kernel_size=1)
        self.init_parameters()

    def forward(self,x):
        '''
        Args:
            x: feature 
        return:
            pi [N x K x A x H x W]
            mu [N x K x A x H x W]
            sigma [N x K x A x H x W]
        '''
        N,H,W = x.size(0),x.size(-2),x.size(-1)
        pi_logit = self.cnn_pi(x).view(N,self.k,self.num_anchors,H,W)
        pi = torch.softmax(pi_logit,dim=1)
        mu = self.cnn_mu(x).view(N,self.k,self.num_anchors,H,W)
        sigma = self.cnn_sigma(x).view(N,self.k,self.num_anchors,H,W)
        if self.sig_max is None:
            sigma = torch.exp(sigma)
        else:
            sigma = (self.sig_max-self.sig_min) *torch.sigmoid(sigma) + self.sig_min
        return {'pi':pi,'mu':mu,'sigma':sigma}
    
    def init_parameters(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # lnit dense
                nn.init.normal_(m.weight,std=0.01)
                nn.init.zeros_(m.bias)