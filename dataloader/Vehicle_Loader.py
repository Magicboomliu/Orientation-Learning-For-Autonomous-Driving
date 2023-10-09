from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset
import os
import sys
sys.path.append("../")
from utils import utils
from skimage import io, transform
import numpy as np
from PIL import Image

from utils.file_io import read_img
from skimage import io, transform


class VerticleLoader(Dataset):
    def __init__(self,datapath,mode,
                 trainlist,vallist,
                 transform=None) -> None:
        super(VerticleLoader,self).__init__()
        
        self.datapath = datapath
        self.mode = mode
        self.transform = transform
        self.trainlist = trainlist
        self.vallist = vallist
        
        verticle_list = {
            "train": self.trainlist,
            "val": self.vallist,
            "test": self.vallist,
        }
        
        
        self.names = ["car_back","car_side","car_front","bus_back","bus_side","bus_front",
                      "truck_back","truck_side","truck_front","motorcycle_back","motorcycle_side","motorcycle_front",
                      "bicycle_back","bicycle_side","bicycle_front"]
        
        
        self.samples = []
        lines = utils.read_text_lines(verticle_list[self.mode])

        for line in lines:
            image = line 
            annotation = image.replace(".jpg",".txt")
            sample = dict()

            sample['image'] = os.path.join(datapath, image)
            sample['annotations'] = os.path.join(datapath, annotation)

            self.samples.append(sample)
        
        
    
    def __getitem__(self, index):
        
        sample = {}
        sample_path = self.samples[index]
        
        sample['image'] = read_img(sample_path['image'])
        sample['annotations'] = np.loadtxt(sample_path['annotations'],dtype=str).reshape(-1,5).astype(np.float)
        
        # only consider cars
        classes = sample["annotations"][:,0]
        nums_of_instances = len(classes)
        cars_inside_list = np.zeros((nums_of_instances,))
        for idx, cls in enumerate(classes):
            if cls in [0,1,2]:
                cars_inside_list[idx] =1.0
        cars_inside_list = cars_inside_list.astype(np.bool)
        sample['annotations'] = sample['annotations'][cars_inside_list]
        assert sample['annotations'].shape[0]!=0
        
        height,width = sample['image'].shape[:2]
        
        
        bbox2d = sample['annotations'][:,1:]
        scale = np.array([width,height,width,height]).reshape(1,4)
        bbox2d = bbox2d * scale # [<x_center> <y_center> <width> <height>]
        # converted to [x_min,y_min,x_max,y_max]
        new_format_bbox2d = np.zeros_like(bbox2d)
        new_format_bbox2d[:,0] = bbox2d[:,0]- bbox2d[:,2]//2 # x_min
        new_format_bbox2d[:,1] = bbox2d[:,1] - bbox2d[:,3]//2 #y_min
        new_format_bbox2d[:,2] = bbox2d[:,0] + bbox2d[:,2]//2 # x-max
        new_format_bbox2d[:,3] = bbox2d[:,1] + bbox2d[:,3]//2 #y_max
        sample['bbox2d'] = new_format_bbox2d
        
        
        #cropped resize image
        cropped_resized_rois = []
        # Image Cropping
        for box in new_format_bbox2d:
            x_min,y_min,x_max,y_max = box
            x_min = int(x_min)
            y_min = int(y_min)
            x_max = int(x_max)
            y_max = int(y_max)
            cropped_roi = sample['image'][y_min:y_max,x_min:x_max]
            cropped_roi_resize = transform.resize(cropped_roi, [224,224], preserve_range=True)
            cropped_resized_rois.append(cropped_roi_resize)
            
        cropped_resized_rois = np.array(cropped_resized_rois)
        sample['cropped_rois'] = cropped_resized_rois
        
        labels = sample["annotations"][:,0]
        sample['labels'] = labels
        assert len(sample['labels']) == len(sample['cropped_rois']) == len(sample['bbox2d'])
        
        
        #random sample
        random_ind = np.random.randint(0,len(sample['bbox2d']),size=3)
        sample['bbox2d'] = sample['bbox2d'][random_ind]
        sample['cropped_rois'] = sample['cropped_rois'][random_ind]
        sample['labels'] = sample['labels'][random_ind]
        sample['annotations'] = sample['annotations'][random_ind]
        
        del sample['annotations']
        
        

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    
    def __len__(self):
        return len(self.samples)

