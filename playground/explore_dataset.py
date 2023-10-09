import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from dataloader.Vehicle_Loader import VerticleLoader
from dataloader import transforms
import cv2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

def draw2Dbbox(image,bbox2d,color=[0,255,0]):
    base_image = image.copy()
    nums_samples = len(bbox2d)
    for idx, bbox in enumerate(bbox2d):
        cv2.rectangle(
            base_image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
           color=color,thickness=2)
    return base_image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def RoI_Normalization(cropped_Rois):
    '''
    cropped_Rois shape: [N,3,H,W]
    '''
    image_mean = torch.from_numpy(np.array(IMAGENET_MEAN)).type_as(cropped_Rois).view(1,3,1,1)
    image_std = torch.from_numpy(np.array(IMAGENET_STD)).type_as(cropped_Rois).view(1,3,1,1)
    
    cropped_Rois = (cropped_Rois - image_mean)/image_std
    
    return cropped_Rois

def RoI_DeNormalization(cropped_Rois):
    '''
    cropped_Rois shape: [N,3,H,W]
    '''
    image_mean = torch.from_numpy(np.array(IMAGENET_MEAN)).type_as(cropped_Rois).view(1,3,1,1)
    image_std = torch.from_numpy(np.array(IMAGENET_STD)).type_as(cropped_Rois).view(1,3,1,1)
    
    cropped_Rois = (cropped_Rois * image_std) + image_mean
    
    return cropped_Rois


from models.resnet import VerticleOrientationNet

if __name__=="__main__":
    datapath = "/data1/liu/OrientationLearning"
    trainlist = "../filenames/cars/train_list_cars.txt"
    vallist = "../filenames/cars/val_list_cars.txt"
    
    
    train_transform_list = [transforms.ToTensor()]
    train_transform = transforms.Compose(train_transform_list)
    
    verticleloader= VerticleLoader(datapath=datapath,mode='train',
                                   trainlist=trainlist,vallist=vallist,transform=train_transform)
   
    train_loader = DataLoader(verticleloader, batch_size =1, \
                                shuffle = True, num_workers = 4, \
                                pin_memory = True)

   
    car_list = ["car_back","car_side","car_front","bus_back","bus_side","bus_front",
                      "truck_back","truck_side","truck_front","motorcycle_back","motorcycle_side","motorcycle_front",
                      "bicycle_back","bicycle_side","bicycle_front"]
    
    model = VerticleOrientationNet(num_layers=34,pretrained=True,num_classes=3).cuda()
    criterion = nn.CrossEntropyLoss()
    
    print(len(train_loader))
    
    
    # for idx, sample in enumerate(train_loader):
        
    #     image = sample['image'] # [N,3,H,W]
    #     bbox2d = sample['bbox2d'] #[N,3,4]
    #     cropped_rois = sample['cropped_rois'] #[N,3,3,224,224]
    #     labels = sample['labels'] #[N,3]
    #     batch_size = labels.shape[0]
    #     labels= labels.reshape(batch_size*3,).to(torch.int64).cuda()
    #     cropped_rois = cropped_rois.reshape(-1,3,224,224).float()

    #     cropped_rois = RoI_Normalization(cropped_Rois=cropped_rois)
        
    #     outputs = model(cropped_rois.cuda())
    #     loss = criterion(outputs,labels.cuda())

    #     _, predicted = outputs.max(1)
        
    #     print(predicted.eq(labels).sum()*1.0/len(predicted))
        
    #     break
        
