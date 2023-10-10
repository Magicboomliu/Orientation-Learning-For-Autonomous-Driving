import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import torchvision
import torchvision.models as models



class VerticleOrientationNet(nn.Module):
    def __init__(self,
                 num_layers=18,pretrained=True,
                 num_classes=1000) -> None:
        super(VerticleOrientationNet,self).__init__()
        
        self.num_classes = num_classes
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        res_feat_chs = {18: 256,
                        34: 256,
                        50: 1024}
        
        self.res_feat_chs = res_feat_chs[num_layers]
        
        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
            
            
        self.regression_head = nn.Linear(self.res_feat_chs * 7 * 7,self.num_classes)

    
    def forward(self,input_image):
        self.features = []
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        
        last_feat = self.features[-1]
        out = F.avg_pool2d(last_feat, 2)
        out = out.contiguous()
        out = out.view(-1, self.res_feat_chs * 7 * 7)
        
        out = self.regression_head(out)
        return out
        
        

if __name__=="__main__":
    
    inputs = torch.randn(10,3,224,224).cuda()
    
    model = VerticleOrientationNet(num_layers=34,pretrained=True,num_classes=3).cuda()
    
    output= model(inputs)
    
    print(output.shape)

