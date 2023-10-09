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

import torch.distributed as dist

# IMAGENET NORMALIZATION
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def RoI_Normalization(cropped_Rois):
    '''cropped_Rois shape: [N,3,H,W]'''
    image_mean = torch.from_numpy(np.array(IMAGENET_MEAN)).type_as(cropped_Rois).view(1,3,1,1)
    image_std = torch.from_numpy(np.array(IMAGENET_STD)).type_as(cropped_Rois).view(1,3,1,1)
    
    cropped_Rois = (cropped_Rois - image_mean)/image_std
    
    return cropped_Rois

def RoI_DeNormalization(cropped_Rois):
    '''cropped_Rois shape: [N,3,H,W]'''
    image_mean = torch.from_numpy(np.array(IMAGENET_MEAN)).type_as(cropped_Rois).view(1,3,1,1)
    image_std = torch.from_numpy(np.array(IMAGENET_STD)).type_as(cropped_Rois).view(1,3,1,1)
    
    cropped_Rois = (cropped_Rois * image_std) + image_mean
    
    return cropped_Rois


class Trainer(object):
    def __init__(self,lr,backbone,devices,
                 datapath,
                 trainlist,
                 vallist,
                 batch_size,
                 test_size,
                 local_rank,
                 pretrain=None):
        self.lr = lr
        self.backbone = backbone
        self.devices = devices
        self.devices = [int(item) for item in devices.split(',')]
        ngpu = len(devices)
        self.ngpu = ngpu
        self.pretrain = pretrain
        
        self.datapath = datapath
        self.trainlist = trainlist
        self.vallist = vallist
        
        self.batch_size =  batch_size
        self.test_batch = test_size
        self.local_rank = local_rank
        
        self.initialize()
        
    
    def _prepare_dataset(self):
        
        datathread = 4
        if os.environ.get('datathread') is not None:
            datathread = int(os.environ.get('datathread'))
        
        if self.local_rank == 0:
            logger.info("Use %d processes to load data..." % datathread)
            
        
        
        train_transform_list = [transforms.ToTensor()]
        train_transform = transforms.Compose(train_transform_list)
        
        val_transform_list = [transforms.ToTensor()]
        val_transform = transforms.Compose(val_transform_list)

        train_dataset = VerticleLoader(datapath=self.datapath,mode='train',
                                   trainlist=self.trainlist,vallist=self.vallist,transform=train_transform)
        test_dataset = VerticleLoader(datapath=self.datapath,mode='val',
                                   trainlist=self.trainlist,vallist=self.vallist,transform=val_transform)


        # define the train sampler for distributed training
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, \
                                        pin_memory=True, num_workers=datathread, sampler=self.train_sampler)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, \
                                        pin_memory=True, num_workers=datathread, sampler=self.test_sampler)
        self.num_batches_per_epoch = len(self.train_loader)
        
        
    def _build_net(self):
        
        self.net = VerticleOrientationNet(num_classes=3,pretrained=True,
                                          num_layers=self.backbone)
        self.is_pretrain = False
        
        
        # loaded the model by distributed model
        device = torch.device("cuda", self.local_rank)
        self.net.cuda(device)
        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[self.local_rank],find_unused_parameters=True)
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.net.parameters()])))
        
        
        if self.pretrain == 'none':
            if self.local_rank==0:
                logger.info('Initial a new model...')
        else:
            if os.path.isfile(self.pretrain):
                model_data = torch.load(self.pretrain)
                logger.info('Load pretrain model: %s', self.pretrain)
                if 'state_dict' in model_data.keys():
                    self.net.load_state_dict(model_data['state_dict'])
                else:
                    self.net.load_state_dict(model_data)
                self.is_pretrain = True
            else:
                logger.warning('Can not find the specific model %s, initial a new model...', self.pretrain)



                
    def _build_optimizer(self):
        beta = 0.999
        momentum = 0.9
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), self.lr,
                                        betas=(momentum, beta), amsgrad=True)

    def _set_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def adjust_learning_rate(self, epoch):
        if epoch>=0 and epoch<=10:
            cur_lr = 1e-4
        elif epoch > 10 and epoch<20:
            cur_lr = 1e-4
        elif epoch>=20 and epoch<30:
            cur_lr = 5e-5
        elif epoch>=30 and epoch<40:
            cur_lr = 3e-5
        elif epoch>=40:
            cur_lr =1.5e-5
        else:
            cur_lr = self.lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
        self.current_lr = cur_lr
        return cur_lr
    
    

    def initialize(self):
        
        # Specific the backen
        dist.init_process_group('nccl',world_size=self.ngpu,rank=self.local_rank)
        # distrubute the GPU: equals CUDA_VISIBLE_DEVICES
        torch.cuda.set_device(self.local_rank)
        
        if self.local_rank==0:
            logger.info(">> Training with distributed parallel.............")
        
        self._prepare_dataset()
        self._build_net()
        self._build_optimizer()
        self._set_criterion()
        
    
    def train_one_epoch(self, epoch, round,iterations,summary_writer,local_rank):
        
        accurate_train_mini_batch_meter = AverageMeter()
        losses_Meter = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # non-detection
        torch.autograd.set_detect_anomaly(True)
        
        # switch to train mode
        self.net.train()
        end = time.time()
        cur_lr = self.adjust_learning_rate(epoch)
        
        if local_rank==0:
            logger.info("learning rate of epoch %d: %f." % (epoch, cur_lr))


        summary_writer.add_scalar("Learning_Rate",cur_lr,epoch+1)
        
        self.train_sampler.set_epoch(epoch)
        
        for i_batch, sample_batched in enumerate(self.train_loader):
            cropped_Rois = torch.autograd.Variable(sample_batched['cropped_rois'].cuda(local_rank), requires_grad=False)
            labels = torch.autograd.Variable(sample_batched['labels'].cuda(local_rank), requires_grad=False)
            
            batch_size = labels.shape[0]
            labels= labels.reshape(batch_size*3,).to(torch.int64)
            cropped_Rois = cropped_Rois.reshape(-1,3,224,224).float()
            cropped_Rois = RoI_Normalization(cropped_Rois)
        
    
            data_time.update(time.time() - end)
            self.optimizer.zero_grad()
            
            # network here
            outputs = self.net(cropped_Rois)
            
            # loss function here
            loss = self.criterion(outputs,labels)
            if type(loss) is list or type(loss) is tuple:
                loss = np.sum(loss)
            
            losses_Meter.update(loss.data.item(), cropped_Rois.size(0))

            _,predicted = outputs.max(1)
            acc_rate_min_batch = predicted.eq(labels).sum()*1.0/len(predicted)
            accurate_train_mini_batch_meter.update(acc_rate_min_batch.data.item(),1)


            # update
            with torch.autograd.detect_anomaly():
                loss.backward()
            self.optimizer.step()
            iterations = iterations+1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if self.local_rank==0:
                if i_batch % 10 == 0:
                    logger.info('this is round %d', round)
                    logger.info('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Acc_Rate {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'.format(
                    epoch, i_batch, self.num_batches_per_epoch, 
                    batch_time=batch_time,
                    data_time=data_time, loss=losses_Meter,flow2_EPE=accurate_train_mini_batch_meter))
            
            
        return losses_Meter.avg,accurate_train_mini_batch_meter.avg,iterations
    
    
    def validate(self,summary_writer,epoch,vis=False,local_rank=0):
        
        batch_time = AverageMeter()
        acc_rate_meter = AverageMeter()
        
        self.net.eval()
        end = time.time()

   
        
        correct = 0
        total = 0
        
        for i, sample_batched in enumerate(self.test_loader):
            cropped_Rois = torch.autograd.Variable(sample_batched['cropped_rois'].cuda(local_rank), requires_grad=False)
            labels = torch.autograd.Variable(sample_batched['labels'].cuda(local_rank), requires_grad=False)
            
            batch_size = labels.shape[0]
            labels= labels.reshape(batch_size*3,).to(torch.int64)
            cropped_Rois = cropped_Rois.reshape(-1,3,224,224).float()
            cropped_Rois = RoI_Normalization(cropped_Rois)
            
            with torch.no_grad():
                outputs = self.net(cropped_Rois)
                
                _,predicted = outputs.max(1)
                
                correct+= predicted.eq(labels).sum()*1.0
                total+=len(predicted)
                
                acc_rate_min_batch = predicted.eq(labels).sum()*1.0/len(predicted)
                acc_rate_meter.update(acc_rate_min_batch.data.item(),1)

            if self.local_rank==0:
                if i % 10 == 0:
                    logger.info('Test: [{0}/{1}]\t Time {2}\t Acc {3}'
                        .format(i, len(self.test_loader), batch_time.val, acc_rate_meter.val))   
                
                
        
        acc = 100 * correct*1.0 /total
        if self.local_rank==0:
            logger.info(' * ACC {:.3f}'.format(acc))
        
        return acc

    def get_model(self):
        return self.net.state_dict()
