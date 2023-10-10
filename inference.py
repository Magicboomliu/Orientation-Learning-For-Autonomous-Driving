import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.AverageMeter import AverageMeter
from utils.common import logger,check_path,count_parameters

from dataloader.Vehicle_Loader import VerticleLoader
from dataloader import transforms
import torch.optim as optim
from models.resnet import VerticleOrientationNet
import os
import time
from utils.file_io import read_img
import pycocotools.mask
import json
from skimage import io, transform
import cv2
import matplotlib.pyplot as plt
import math

# IMAGENET NORMALIZATION
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

def load_network(nums_layers =34, pretrained_path = None):
    pretrained_model = VerticleOrientationNet(num_classes=3,pretrained=True,
                                          num_layers=nums_layers)
    pretrained_model = torch.nn.DataParallel(pretrained_model, device_ids=[0]).cuda()
    model_data = torch.load(pretrained_path)
    if 'state_dict' in model_data.keys():
        pretrained_model.load_state_dict(model_data['state_dict'])
    else:
        pretrained_model.load_state_dict(model_data)

    print("Loaded the Pre-trained Model Successufully at:  {}".format(pretrained_path))

    return pretrained_model

def read_annotation(annotation_filename,class_names =["N/A",
                    "car"]):
    with open(annotation_filename) as file:
        annotation = json.load(file)
    intrinsic_matrix = torch.as_tensor(annotation["intrinsic_matrix"])
    extrinsic_matrix = torch.as_tensor(annotation["extrinsic_matrix"])
    instance_ids = {
        class_name: list(masks.keys())
        for class_name, masks in annotation["masks"].items()
        if class_name in class_names
    }

    # if contains the instances ids
    if instance_ids:
        masks = torch.cat([
            torch.stack([
                torch.as_tensor(
                    data=pycocotools.mask.decode(annotation["masks"][class_name][instance_id]),
                    dtype=torch.float,
                )
                for instance_id in instance_ids
            ], dim=0)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        labels = torch.cat([
            torch.as_tensor(
                data=[class_names.index(class_name)] *  len(instance_ids),
                dtype=torch.long,
            )
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        boxes_3d = torch.cat([
            torch.stack([
                torch.as_tensor(
                    data=annotation["boxes_3d"][class_name].get(instance_id, [[np.nan] * 3] * 8),
                    dtype=torch.float,
                )
                for instance_id in instance_ids
            ], dim=0)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        instance_ids = torch.cat([
            torch.as_tensor(
                data=list(map(int, instance_ids)),
                dtype=torch.long,
            )
            for instance_ids in instance_ids.values()
        ], dim=0)

        return dict(
            masks=masks,
            labels=labels,
            boxes_3d=boxes_3d,
            instance_ids=instance_ids,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
        )
    
    # else returan 
    else:
        return dict(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
        )

def get_2D_bounding_boxes(instance_mask):
    '''
    Inputs:
    instance mask shape : [N,H,W]
    
    Outputs:
    bounding boxes shape: [N,4]
    '''
    nums_instances, height, width = instance_mask.shape
    bounding_boxes = []
    
    for i in range(nums_instances):
        # get the current instance mask
        mask = instance_mask[i]
        non_zero_pixels = np.where(mask!=0)
        if non_zero_pixels[0].size>0:
            min_x = np.min(non_zero_pixels[1])
            max_x = np.max(non_zero_pixels[1])
            min_y = np.min(non_zero_pixels[0])
            max_y = np.max(non_zero_pixels[0])
            if max_y<=min_y:
                max_y = min_y+1
            if max_x<=min_x:
                max_x = min_x+1
            
            # add to the bounding box list
            bounding_boxes.append([min_x,min_y,max_x,max_y])

        else:
            bounding_boxes.append([0,0,0,0])
    
    return np.array(bounding_boxes)

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

CATGORIES = ["car_back","car_side","car_front"]
import argparse



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', type=str, help='pretrained_models', default='pretrained_models/model_best.pth')
    parser.add_argument('--nums_layers', type=int, default=34, help='input batch size')
    parser.add_argument('--image_path', type=str, help='image_path', default='input_example/image/0000000954.png')
    parser.add_argument('--annotation_path', type=str, help='image_path', default='input_example/annotations/json/0000000954.json')
    parser.add_argument('--saved_folder', type=str, help='image_path', default='outputs_vis')
    

    # pretrained_path = "pretrained_models/model_best.pth"
    opt = parser.parse_args()
    if not os.path.exists(opt.saved_folder):
        os.makedirs(opt.saved_folder)

    pretrained_model = load_network(nums_layers=34,pretrained_path=opt.pretrained_model_path)
    left_image_data = read_img(opt.image_path)
    seg_mask = read_annotation(opt.annotation_path)['masks']
    bbox2d = get_2D_bounding_boxes(seg_mask) # (N,4)


    cropped_resized_rois = []
    
    # Image Cropping to get Cropped Rois
    for box in bbox2d:
        x_min,y_min,x_max,y_max = box
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)
        cropped_roi = left_image_data[y_min:y_max,x_min:x_max]
        cropped_roi_resize = transform.resize(cropped_roi, [224,224], preserve_range=True)
        cropped_resized_rois.append(cropped_roi_resize)
        
    cropped_resized_rois = np.array(cropped_resized_rois) # (N,H,W,3)

    roi_images = np.transpose(cropped_resized_rois,(0,3,1,2))
    roi_images = torch.from_numpy(roi_images)/255.
    roi_images = RoI_Normalization(roi_images)
    
    with torch.no_grad():
        outputs = pretrained_model(roi_images)
        _,predicted = outputs.max(1)
        estimated_labels = []
        for idx, pred in  enumerate(predicted):
            estimated_labels.append(CATGORIES[pred])
        

    

    # visualization
    images_with_labels = draw2Dbbox(image=left_image_data,bbox2d=bbox2d,color=[0,255,0])
    plt.axis("off")
    plt.imshow(images_with_labels/255)
    plt.savefig("{}/imageswith2d_{}.png".format(opt.saved_folder,os.path.basename(opt.image_path)),bbox_inches='tight')

    if cropped_resized_rois.shape[0]*1.0> int(math.sqrt(cropped_resized_rois.shape[0])):
        nums_column = int(math.sqrt(cropped_resized_rois.shape[0]))+1   
    for i in range(nums_column**2):
        if i>=cropped_resized_rois.shape[0]:
            break

        plt.subplot(nums_column,nums_column,i+1)
        plt.title(estimated_labels[i])
        plt.axis("off")
        plt.imshow(cropped_resized_rois[i]/255)
        
    plt.savefig("{}/Estimated_ROI_Labels-{}.png".format(opt.saved_folder,os.path.basename(opt.image_path)),bbox_inches='tight')
        
    







    pass