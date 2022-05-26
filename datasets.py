#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:45:48 2020

A python implementation of CIGMO (Categorical Invariant Generative MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""

import numpy as np
import random
import torch
import torch.utils.data
from torchvision import transforms
import os
from tqdm import tqdm

class DatasetFromTensor(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        if split == 'train':
            self.data = torch.load(os.path.join(root, 'trainset.pt'))
        elif split == 'test':
            self.data = torch.load(os.path.join(root, 'testset.pt'))
        elif split == 'val':
            self.data = torch.load(os.path.join(root, 'valset.pt'))
        else:
            raise Exception('unsupported split: ' + split)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        (img, label, _) = self.data[index]

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def get_attr(self, index):
        (_, _, attr) = self.data[index]        
        return attr
    
class SubsetWithAttributes(torch.utils.data.Subset):
    def get_attr(self, index):
        return self.dataset.get_attr(self.indices[index])

class DatasetGrouped(torch.utils.data.Dataset):
    def __init__(self, base_dataset, group_size):

        self.base_dataset = base_dataset    
        self.data_dict = {}
        self.group_size = group_size
        for i in tqdm(range(self.__len__()), position=0, leave=True, disable=False):
            image, label = self.base_dataset.__getitem__(i)
            try:
                self.data_dict[label]
            except KeyError:
                self.data_dict[label] = []
            self.data_dict[label].append(image)
            
    def __len__(self):
        return self.base_dataset.__len__()

    def __getitem__(self, index):
        image, label = self.base_dataset.__getitem__(index)
        group = [image]
        for i in range(self.group_size-1):
            group.append(random.choice(self.data_dict[label]))
        group = torch.stack(group, dim=0)
        return group, label
    
    def getAllInstanceOfLabel(self, label):
        return self.data_dict[label]
        
    def getLabels(self):
        return list(self.data_dict.keys())
    

def shapenet_standard_subset(num_class):
    # target_classes = [
    #      '02958343',    # car
    #      '03001627',    # chair
    #      '04379243',    # table
    #      '02691156',    # airplane
    #      '04256520',    # sofa
    #      '04090263',    # rifle
    #      '03636649',    # lamp
    #      '04530566',    # boat
    #      '02828884',    # bench
    #      '03691459',    # box
    #      '04468005',    # train
    #      ]    
    target_classes = [
         '02958343',    # car
         '03001627',    # chair
         '04379243',    # table
         '02691156',    # airplane
#         '04256520',    # sofa
#         '04090263',    # rifle
         '03636649',    # lamp
         '04530566',    # boat
#         '02828884',    # bench
         '03691459',    # box
#         '02933112',    # cabin
         '03211117',    # display
#         '04401088',    # cell phone
         '02924116',    # truck
#         '02808440',    # bath tub
#         '03467517',    # guitar
#         '03325088',    # water tap
#         '03046257',    # clock
#         '03991062',    # vase 1
         '03593526',    # vase 2
         '02876657',    # bottle
         '03642806',    # easy chair
         '02871439',    # shelf
#         '03624134',    # knife
         '04468005',    # train
         ]    
    subset = target_classes[:num_class]    
    class_dict = dict()
    for i in range(len(subset)):
        class_dict[subset[i]] = i
    return subset, class_dict
    
    
def load_shapenet_dataset(root, subset=None, split='train'):
    tran = transforms.Compose(
    #     ([ transforms.Grayscale() ] if not color else []) +
         [
          transforms.ToTensor(),
          ])
    dataset = DatasetFromTensor(root=root, split=split, transform=tran)
    if subset != None:
        idx = list(filter(lambda i: dataset[i][1][0] in subset, range(len(dataset))))
        dataset = torch.utils.data.Subset(dataset, idx)
    return dataset
    
def mvc_standard_subset(ds):
    exclusion_keys = [    
        'Bandeau',
        'BoxerBriefs',
        'Boxers',
        'BoyShorts',
        'Bras',
        'Brief',
        'Crossback',
        'Lingerie',
        'Panties',
        'Hosiery',
        'OnePieceSwimsuits',
        'SwimSets',
        'SwimsuitBottoms',
        'SwimsuitTops']
    idx = list(filter(lambda i: all([ds.get_attr(i)[k] == 0 for k in exclusion_keys]), 
                      range(len(ds))))
    return SubsetWithAttributes(ds, idx)

    
    