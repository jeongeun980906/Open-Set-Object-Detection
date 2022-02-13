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

class MixtureOfLogits(nn.Module):
    def __init__(self,
                 in_dim     = 64,   # input feature dimension 
                 y_dim      = 10,   # number of classes 
                 k          = 5,    # number of mixtures
                 sig_min    = 1, # minimum sigma
                 sig_max    = 10, # maximum sigma
                 SHARE_SIG  = True  # share sigma among mixture
                 ):
        super(MixtureOfLogits,self).__init__()
        self.in_dim     = in_dim    # Q
        self.y_dim      = y_dim     # D
        self.k          = k         # K
        self.sig_min    = sig_min
        self.sig_max    = sig_max
        self.SHARE_SIG  = SHARE_SIG
        self.build_graph()

    def build_graph(self):
        self.fc_pi      = nn.Linear(self.in_dim,self.k)
        self.fc_mu      = nn.Linear(self.in_dim,self.k*self.y_dim)
        if self.SHARE_SIG:
            self.fc_sigma   = nn.Linear(self.in_dim,self.k)
        else:
            self.fc_sigma   = nn.Linear(self.in_dim,self.k*self.y_dim)

    def forward(self,x):
        """
            :param x: [N x Q]
        """
        pi_logit        = self.fc_pi(x)                                 # [N x K]
        pi              = torch.softmax(pi_logit,dim=1)                 # [N x K]
        mu              = self.fc_mu(x)                                 # [N x KD]
        mu              = torch.reshape(mu,(-1,self.k,self.y_dim))      # [N x K x D]
        if self.SHARE_SIG:
            sigma       = self.fc_sigma(x)                              # [N x K]
            sigma       = sigma.unsqueeze(dim=-1)                       # [N x K x 1]
            sigma       = sigma.expand_as(mu)                           # [N x K x D]
        else:
            sigma       = self.fc_sigma(x)                              # [N x KD]
        sigma           = torch.reshape(sigma,(-1,self.k,self.y_dim))   # [N x K x D]
        if self.sig_max is None:
            sigma = self.sig_min + torch.exp(sigma)                     # [N x K x D]
        else:
            sig_range = (self.sig_max-self.sig_min)
            sigma = self.sig_min + sig_range*torch.sigmoid(sigma)       # [N x K x D]
        mol_out = {'pi':pi,'mu':mu,'sigma':sigma}
        return mol_out

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.normal_(m.weight,std=0.01)
                nn.init.constant_(m.bias, 0)