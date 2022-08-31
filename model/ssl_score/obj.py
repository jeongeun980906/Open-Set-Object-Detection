from torchvision import models
import torchvision
import torch.nn as nn

class simple_cnn(nn.Module):
    def __init__(self,y_dim=1):
        super().__init__()
        model = models.vgg16() 
        self.features = model.features
        self.classifier = nn.Sequential(
            nn.Linear(512*8*8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.cls_fc = nn.Linear(4096, y_dim)
        # self.init_param()

    def forward(self,x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.view(batch_size,-1)
        x = self.classifier(x)
        x = self.cls_fc(x)
        return x