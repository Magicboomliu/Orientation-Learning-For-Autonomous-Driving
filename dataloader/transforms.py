from __future__ import division
from genericpath import samefile
import torch
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as F
import random
import cv2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __call__(self, sample):
        left = np.transpose(sample['image'], (2, 0, 1))  # [3, H, W]
        sample['image'] = torch.from_numpy(left) / 255.
        roi_images = np.transpose(sample['cropped_rois'],(0,3,1,2))
        sample['cropped_rois'] = torch.from_numpy(roi_images)/255.

        return sample

