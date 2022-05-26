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

class MGVAENet(nn.ModuleDict):

    def __init__(self, num_cluster, content_dim, view_dim,
                 color=False, 
                 decoder_last_sigmoid=True, single_view_encoder=False,
                 breadth_ratio=1, adapt_prior=False):
        
        super().__init__()
        
        self.num_cluster = num_cluster
        self.content_dim = content_dim        
        self.view_dim = view_dim        
        self.color = color
        self.breadth_ratio = breadth_ratio
        self.decoder_last_sigmoid = decoder_last_sigmoid
        self.single_view_encoder = single_view_encoder
        self.adapt_prior = adapt_prior
        
        self.nf_conv1 = 32 * self.breadth_ratio
        self.nf_conv2 = 64 * self.breadth_ratio
        self.nf_conv3 = 128 * self.breadth_ratio
        self.nf_fc1 = 500 * self.breadth_ratio
        
        self.info = dict()
        
        if color:
            self.num_channel = 3
        else:
            self.num_channel = 1
        
        self.set_cluster_prior()        
        self.add_classifier()        
        self.add_encoder()          
        self.add_decoder()
        
    def set_cluster_prior(self):
        
        self.cluster_prior = torch.zeros(self.num_cluster)
        self.cluster_prior[:] = 1 / self.num_cluster
        if self.adapt_prior:
            self.cluster_prior = nn.Parameter(self.cluster_prior)

    def add_classifier(self):

        self.classifier = nn.Sequential(
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
            nn.Conv2d(self.nf_fc1, self.num_cluster, kernel_size=(1), bias=True),
            # nn.Softmax(dim=1)
            )

    def add_encoder(self):
        
        self.encoder_common = nn.Sequential(
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
                nn.ReLU(inplace=True)
                )

        self.encoder_content_mu = nn.ModuleList()
        self.encoder_content_sig = nn.ModuleList()
        self.encoder_view_mu = nn.ModuleList()
        self.encoder_view_sig = nn.ModuleList()

        for k in range(self.num_cluster):
            
            self.encoder_content_mu.append(nn.Sequential(
                    nn.Conv2d(self.nf_fc1, self.content_dim, kernel_size=(1), bias=True)
                    ))
            
            self.encoder_content_sig.append(nn.Sequential(
                    nn.Conv2d(self.nf_fc1, self.content_dim, kernel_size=(1), bias=True),
                    nn.Softplus()
                    ))
    
            if k == 0 or not self.single_view_encoder: 
                self.encoder_view_mu.append(nn.Sequential(
                        nn.Conv2d(self.nf_fc1, self.view_dim, kernel_size=(1), bias=True)
                        ))
                
                self.encoder_view_sig.append(nn.Sequential(
                        nn.Conv2d(self.nf_fc1, self.view_dim, kernel_size=(1), bias=True),
                        nn.Softplus()
                        ))
            else:
                self.encoder_view_mu.append(self.encoder_view_mu[0])
                self.encoder_view_sig.append(self.encoder_view_sig[0])
                
            
    def add_decoder(self):
        
        self.decoder_common = nn.Sequential(
                nn.ConvTranspose2d(self.nf_fc1, self.nf_conv3, kernel_size=(8), bias=True),
                nn.BatchNorm2d(self.nf_conv3),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(self.nf_conv3, self.nf_conv2, kernel_size=(6), stride=2, padding=2, bias=True),
                nn.BatchNorm2d(self.nf_conv2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(self.nf_conv2, self.nf_conv1, kernel_size=(6), stride=2, padding=2, bias=True),
                nn.BatchNorm2d(self.nf_conv1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(self.nf_conv1, self.num_channel, kernel_size=(6), stride=2, padding=2, bias=True),
                nn.Sigmoid() if self.decoder_last_sigmoid else nn.Identity()
                )

        self.decoder = nn.ModuleList()

        for k in range(self.num_cluster):

            self.decoder.append(nn.Sequential(
                    nn.ConvTranspose2d(self.content_dim + self.view_dim, self.nf_fc1, kernel_size=(1), bias=True),
                    nn.BatchNorm2d(self.nf_fc1),
                    nn.ReLU(inplace=True),
                    ))
            
def save_net(net, path):
    net.cpu()
    s = { 'num_cluster': net.num_cluster,
          'content_dim': net.content_dim,
          'view_dim': net.view_dim,
          'color': net.color,
          'decoder_last_sigmoid': net.decoder_last_sigmoid,
          'single_view_encoder': net.single_view_encoder,
          'breadth_ratio': net.breadth_ratio,
          'state_dict': net.state_dict(),
          'info': net.info,
          }    
    torch.save(s, path)

def load_net(path):
    s = torch.load(path)
    net = MGVAENet(s['num_cluster'], s['content_dim'], s['view_dim'], 
                   color=s['color'], decoder_last_sigmoid=s['decoder_last_sigmoid'], 
                   single_view_encoder=s['single_view_encoder'], 
                   breadth_ratio=s['breadth_ratio'])
    net.load_state_dict(s['state_dict'])
    net.info = s['info']
    return net

        