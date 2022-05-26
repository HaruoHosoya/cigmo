#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:10:33 2020

A python implementation of CIGMO (Categorical Invariant Generative MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""


import numpy as np
from tqdm import tqdm

import torch

from IID_losses import IID_loss

eps = 1e-20

def train_iic(net, dataloader, optimizer, num_epochs, device='cpu', 
              no_update=False):
        
    net.to(device)    
    net.train()

    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        
        epoch_loss = 0.0
        sd = net.state_dict()
        train_oc = (net.num_overcluster != None) and ((epoch % 2) == 0)

        for inputs, _ in tqdm(dataloader, position=0, leave=True):
            
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            group_size = inputs.size(1)
            
            if not no_update: optimizer.zero_grad()
            
            with torch.set_grad_enabled(not no_update):
                loss = 0

                if train_oc:
                    prob_logits = net.classifier_oc(inputs.reshape(batch_size * group_size, inputs.size(2), inputs.size(3), inputs.size(4))).reshape((batch_size, group_size, -1))
                else:
                    prob_logits = net.classifier(inputs.reshape(batch_size * group_size, inputs.size(2), inputs.size(3), inputs.size(4))).reshape((batch_size, group_size, -1))
                prob = torch.softmax(prob_logits, 2)

                for i in range(group_size-1):
                    for j in range(i+1, group_size):
                        iid_loss, _ = IID_loss(prob[:,i,:].reshape(batch_size, -1), prob[:,j,:].reshape(batch_size, -1))
                        loss += iid_loss
                
                loss = loss / batch_size 
                if np.isnan(loss.item()):
                    print('loss is nan')
                    net.sd = sd
                    return None
                
                if not no_update:
                    loss.backward()
                    optimizer.step()
                    
                epoch_loss += loss.item() * batch_size 
                
        epoch_loss = epoch_loss / len(dataloader.dataset) 
        
 
        if not no_update:
            net.info['train_loss'] = epoch_loss
            net.info['train_num_epochs'] = num_epochs
        
        print('Loss: {:.4f}'.format(epoch_loss))
        
    print('Training done')
    return epoch_loss

