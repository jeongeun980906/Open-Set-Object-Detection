import torch
import numpy as np
from sklearn.cluster import KMeans 
import json
class maha_distance():
    def __init__(self,num_cluster=21):
        self.num_cluster = num_cluster
        self.type = 'RMD'
        self.T = 1000

    def model_feature(self,ftrain):
        ftrain = np.asarray(ftrain)
        self.mu_0 = np.mean(ftrain,axis=0)
        self.sigma_0 = np.cov(ftrain.T)
        print(self.sigma_0.shape)
        kmeans = KMeans(n_clusters=self.num_cluster)
        kmeans.fit(ftrain)
        mu_k = []
        std = np.eye(ftrain.shape[1])
        size = ftrain.shape[0]
        label = kmeans.labels_
        for i in range(self.num_cluster):
            index = np.where(label == i)[0]
            data = ftrain[index]
            mu = np.mean(data,axis=0)
            a = np.expand_dims((data-mu),-1)
            pa = np.transpose(a,(0,2,1))
            # print(a.shape,pa.shape)
            temp = np.matmul(a,pa) # [M x D x D]
            temp = np.sum(temp,axis=0)
            mu_k.append(mu)
            std += temp 
        self.std_k = std/size # [D x D]
        self.mu_k = np.vstack(mu_k) # [K x D]

    def score(self,ftest):
        '''
        ftest [N x D] 
        mu_k [K x D]
        std_k [D x D]
        (ftest - mu_k) std_k^-1 (ftet-mu_k)
        '''
        std = self.std_k
        maha_total = []
        print(self.type)
        for i in range(self.num_cluster):
            mu = self.mu_k[i]
            dis = ftest-mu # [N x D]
            dis = np.expand_dims(dis,axis=1) # [N x 1 x D]
            maha = np.matmul(dis,std) 
            maha = np.matmul(maha,np.transpose(dis,(0,2,1)))[:,0,0] # [N x 1 x 1]
            maha_total.append(maha)
        maha_total = np.vstack(maha_total)
        if self.type == 'MD':
            return np.min(maha_total,axis=0)
        elif self.type == 'RMD':
            dis = ftest-self.mu_0 # [N x D]
            dis = np.expand_dims(dis,axis=1) # [N x 1 x D]
            maha_0 = np.matmul(dis,self.sigma_0) 
            maha_0 = np.matmul(maha_0,np.transpose(dis,(0,2,1)))[:,0,0] # [N x 1 x 1]
            return np.min(maha_total,axis=0) - maha_0
    
    def save(self,path):
        log = {
            'num_clusters': self.num_cluster,
            'mu_k': self.mu_k.tolist(),
            'sigma_k': self.std_k.tolist(),
            'mu_0': self.mu_0.tolist(),
            'sigma_0': self.sigma_0.tolist()
        }
        with open(path, "w") as json_file:
            json.dump(log, json_file,indent=4)

    def load(self,path):
        with open(path, "r") as json_file:
            data = json.load(json_file)
        self.mu_k = np.asarray(data['mu_k'])
        self.std_k = np.asarray(data['sigma_k'])
        self.mu_0 = np.asarray(data['mu_0'])
        self.sigma_0 = np.asarray(data['sigma_0'])