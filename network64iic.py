#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:17:45 2020

A python implementation of CIGMO (Categorical Invariant Generative MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""

import torch
import torch.nn as nn

class IICNet(nn.ModuleDict):

    def __init__(self, num_cluster, num_overcluster=None,
                 color=False, 
                 breadth_ratio=1):
        
        super().__init__()
        
        self.num_cluster = num_cluster
        self.num_overcluster = num_overcluster
        self.color = color
        self.breadth_ratio = breadth_ratio
        
        self.nf_conv1 = 32 * self.breadth_ratio
        self.nf_conv2 = 64 * self.breadth_ratio
        self.nf_conv3 = 128 * self.breadth_ratio
        self.nf_fc1 = 500 * self.breadth_ratio
        
        self.info = dict()
        
        if color:
            self.num_channel = 3
        else:
            self.num_channel = 1
        
        self.add_classifier()        
        
    def add_classifier(self):

        self.classifier_common = nn.Sequential(
            nn.Conv2d(self.num_channel, self.nf_conv1, kernel_size=(5), stride=2, padding=2, bias=True),
            nn.BatchNorm2d(self.nf_conv1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nf_conv1, self.nf_conv2, kernel_size=(5), stride=2, padding=2, bias=True),
            nn.BatchNorm2d(self.nf_conv2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nf_conv2, self.nf_conv3, kernel_size=(5), stride=2, padding=2, bias=True),
            nn.BatchNorm2d(self.nf_conv3),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nf_conv3, self.nf_fc1, kernel_size=(8), bias=True),
            nn.BatchNorm2d(self.nf_fc1),
            nn.ReLU(inplace=True),
            )

        self.classifier = nn.Sequential(
            self.classifier_common,
            nn.Conv2d(self.nf_fc1, self.num_cluster, kernel_size=(1), bias=True),
            # nn.Softmax(dim=1)
            )
           
        if self.num_overcluster != None:
            self.classifier_oc = nn.Sequential(
                self.classifier_common,
                nn.Conv2d(self.nf_fc1, self.num_overcluster, kernel_size=(1), bias=True),
                # nn.Softmax(dim=1)
                )

def save_net(net, path):
    net.cpu()
    s = { 'num_cluster': net.num_cluster,
          'num_overcluster': net.num_overcluster,
          'color': net.color,
          'breadth_ratio': net.breadth_ratio,
          'state_dict': net.state_dict(),
          'info': net.info,
          }    
    torch.save(s, path)

def load_net(path):
    s = torch.load(path)
    net = IICNet(s['num_cluster'], s['num_overcluster'], 
                   color=s['color'], breadth_ratio=s['breadth_ratio'])
    net.load_state_dict(s['state_dict'])
    net.info = s['info']
    return net

        