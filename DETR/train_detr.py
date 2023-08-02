##############################################################################
#
#
#   This code is an adaptation of the DETR algorithm which has been modified
#   to work with an arbitrary number of classes.
#
#
##############################################################################

import os
import numpy as np 
#import pandas as pd 
from datetime import datetime
import time
import random
from tqdm.autonotebook import tqdm
from argparse import ArgumentParser


#Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

# Dataloader
from dataset import CustomImageDataset

################# DETR FUCNTIONS FOR LOSS######################## 
import sys
'''
print(sys.path)
sys.path.append('./detr/')  # The python interpreter is dumb so we need to change directories
sys.path.append('./detr/util/')
print(sys.path)
'''

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
#################################################################

#Albumenatations
#import albumentations as A
import matplotlib.pyplot as plt
#from albumentations.pytorch.transforms import ToTensorV2

#Glob
from glob import glob

'''
DETR model
    - num_classes:
        number of classes. In this example will always be 2 as we can either have a penguin
        or turtle.
    - num_queries:
        number of objects we want to identify in the image. In this case always 1 as there
        is only one animal in each image.
'''
class DETRModel(nn.Module):
    def __init__(self,num_classes,num_queries):
        super(DETRModel,self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.in_features = self.model.class_embed.in_features
        
        self.model.class_embed = nn.Linear(in_features=self.in_features,out_features=self.num_classes)
        self.model.num_queries = self.num_queries
        
    def forward(self,images):
        return self.model(images)

'''
AverageMeter
- Class for averaging loss, metric, etc over epochs
'''
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# CONFIG
matcher = HungarianMatcher()

weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}

losses = ['labels', 'boxes']

n_folds = 5
seed = 42
num_classes = 3 # Penguin or turtle and background
num_queries = 2 # Only one in image but if there are multiple we can increment this
null_class_coef = 0.5
BATCH_SIZE = 16
LR = 2e-6
EPOCHS = 500

def train_fn(data_loader,model,criterion,optimizer,device,scheduler,epoch):
    model.train()
    criterion.train()
    
    summary_loss = AverageMeter()
    
    tk0 = tqdm(data_loader, total=len(data_loader))     # An iterable that also progresses a progress bar
    
    for step, (images, targets) in enumerate(tk0):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        output = model(images)

        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        summary_loss.update(losses.item(),BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)
        
    return summary_loss

def eval_fn(data_loader, model,criterion, device):
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()
    
    with torch.no_grad():
        
        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets) in enumerate(tk0):
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)
        
            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict
        
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            summary_loss.update(losses.item(),BATCH_SIZE)
            tk0.set_postfix(loss=summary_loss.avg)
    
    return summary_loss

def collate_fn(batch):
    return tuple(zip(*batch))

def run(args, device, train_dataset, valid_dataset):
    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=args.batch, 
                                    shuffle=True,
                                    collate_fn=collate_fn)

    valid_dataloader = DataLoader(valid_dataset, 
                                    batch_size=args.batch, 
                                    shuffle=True,
                                    collate_fn=collate_fn)
    
    device = torch.device(device)
    model = DETRModel(num_classes=num_classes,num_queries=num_queries)
    model = model.to(device)
    criterion = SetCriterion(num_classes-1, matcher, weight_dict, eos_coef = null_class_coef, losses=losses)
    criterion = criterion.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    train_losses = []
    valid_losses = []
    best_loss = 10**5
    for epoch in range(EPOCHS):
        train_loss = train_fn(train_dataloader, model,criterion, optimizer,device,scheduler=None,epoch=epoch)
        valid_loss = eval_fn(valid_dataloader, model,criterion, device)
        
        print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch+1,train_loss.avg,valid_loss.avg))
        train_losses.append(train_loss.avg)
        valid_losses.append(valid_loss.avg)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print('Best model found in Epoch {}........Saving Model'.format(epoch+1))
            torch.save(model.state_dict(), f'detr_best_{epoch}.pth')
    
    x = np.arange(1, EPOCHS + 1)
    fig, ax = plt.subplots()
    ax.plot(x, train_losses, label="train loss")
    ax.plot(x, valid_losses, label="validation loss")
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.35),
        ncol=3, 
    )
    plt.show()

    return model