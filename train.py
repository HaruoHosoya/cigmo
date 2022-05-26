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

from IID_losses import IID_loss

import torch

eps = 1e-20

def train_model(net, dataloader, optimizer, num_epochs, device='cpu', 
                k_mse=1, k_kldiv_cluster=1, k_kldiv_latent=1,
                a_posterior='average', no_update=False):
        
    net.to(device)
    
    net.train()
    
    eps1 = eps if a_posterior != 'product' else 1e-8

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        
        epoch_loss = 0.0
        sd = net.state_dict()
        
        for inputs, _ in tqdm(dataloader, position=0, leave=True):
            
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            group_size = inputs.size(1)
            
            if not no_update: optimizer.zero_grad()
            
            with torch.set_grad_enabled(not no_update):
                loss = 0

                cluster_posterior_logits = net.classifier(inputs.reshape(batch_size * group_size, inputs.size(2), inputs.size(3), inputs.size(4))).reshape((batch_size, group_size, net.num_cluster))
                if a_posterior == 'average':
                    cluster_posterior = torch.mean(torch.softmax(cluster_posterior_logits, 2), 1)
                elif a_posterior == 'logit_average':
                    cluster_posterior = torch.softmax(torch.mean(cluster_posterior_logits, 1), 1)
                elif a_posterior == 'product':
                    cluster_posterior = torch.prod(torch.softmax(cluster_posterior_logits, 2), 1)
                    cluster_posterior = cluster_posterior / torch.sum(cluster_posterior, 1, keepdim=True)                    

                prior = net.cluster_prior.to(device).softmax(dim=0)
                loss += k_kldiv_cluster * torch.sum(cluster_posterior * torch.log(cluster_posterior / prior + eps1), (0, 1))

                if torch.isnan(loss).any():
                    print('loss (kl-class) is nan')
                    
                interm = []
                for k in range(group_size):
                    interm.append(net.encoder_common(inputs[:,k]))                    

                for c in range(net.num_cluster):
                    content_mu = []
                    content_sig = []
                    view_mu = []
                    view_sig = []
                    for k in range(group_size):
                        content_mu.append(net.encoder_content_mu[c](interm[k]))
                        content_sig.append(net.encoder_content_sig[c](interm[k]))
                        view_mu.append(net.encoder_view_mu[c](interm[k]))
                        view_sig.append(net.encoder_view_sig[c](interm[k]))
                    
                    content_mu = torch.mean(torch.stack(content_mu, dim=0), dim=0)
                    content_sig = torch.sqrt(torch.mean(torch.stack(content_sig, dim=0) ** 2, dim=0))
                    
                    kl_content = torch.sum(kldiv_gaussian(content_mu, content_sig), (1, 2, 3))
                    loss += -k_kldiv_latent * torch.sum(cluster_posterior[:, c] * kl_content, 0)
    
                    if torch.isnan(loss).any():
                        print('loss (kl) is nan (cluster={}, content)'.format(c))
                    
                    for k in range(group_size):
                        kl_view = torch.sum(kldiv_gaussian(view_mu[k], view_sig[k]), (1, 2, 3))
                        loss += -k_kldiv_latent * torch.sum(cluster_posterior[:, c] * kl_view, 0)
                        
                        if torch.isnan(loss).any():
                            print('loss (kl) is nan (cluster={}, view={})'.format(c, k))
    
                    content_latent = add_noise(content_mu, content_sig)
                    
                    for k in range(group_size):
                        view_latent = add_noise(view_mu[k], view_sig[k])                    
                        latent = torch.cat((content_latent, view_latent), dim=1)
                        recons = net.decoder_common(net.decoder[c](latent))
                        mse = 0.5 * torch.sum((recons - inputs[:,k]) ** 2 + 0.5 * np.log(2 * np.pi), (1, 2, 3))
                        loss += k_mse * torch.sum(cluster_posterior[:, c] * mse, 0)
    
                        if torch.isnan(loss).any():
                            print('loss (MSE) is nan (cluster={})'.format(c))
  
                
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
            net.info['train_k_mse'] = k_mse
            net.info['train_k_kldiv_cluster'] = k_kldiv_cluster
            net.info['train_k_kldiv_latent'] = k_kldiv_latent
            net.info['train_a_posterior'] = a_posterior
        
        print('Loss: {:.4f}'.format(epoch_loss))
        
    print('Training done')
    return epoch_loss


def train_model_mlvae(net, dataloader, optimizer, num_epochs, device='cpu', 
                      k_mse=1, k_kldiv_cluster=1, k_kldiv_latent=1,
                      a_posterior='average', no_update=False, k_iic=0):
        
    net.to(device)
    
    net.train()
    
    eps1 = eps if a_posterior != 'product' else 1e-10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        
        epoch_loss = 0.0
        sd = net.state_dict()

        for inputs, _ in tqdm(dataloader, position=0, leave=True):
            
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            group_size = inputs.size(1)
            
            if not no_update: optimizer.zero_grad()
            
            with torch.set_grad_enabled(not no_update):
                loss = 0

                cluster_posterior_logits = net.classifier(inputs.reshape(batch_size * group_size, inputs.size(2), inputs.size(3), inputs.size(4))).reshape((batch_size, group_size, net.num_cluster))
                if a_posterior == 'average':
                    cluster_posterior = torch.mean(torch.softmax(cluster_posterior_logits, 2), 1)
                elif a_posterior == 'logit_average':
                    cluster_posterior = torch.softmax(torch.mean(cluster_posterior_logits, 1), 1)
                elif a_posterior == 'product':
                    cluster_posterior = torch.prod(torch.softmax(cluster_posterior_logits, 2), 1)
                    cluster_posterior = cluster_posterior / torch.sum(cluster_posterior, 1, keepdim=True)                    

                loss += k_kldiv_cluster * torch.sum(cluster_posterior * torch.log(cluster_posterior / net.cluster_prior.to(device) + eps1), (0, 1))

                if torch.isnan(loss).any():
                    print('loss (kl-class) is nan')
                    
                if k_iic > 0:
                    prob = torch.softmax(cluster_posterior_logits, 2)
                    for i in range(group_size-1):
                        for j in range(i+1, group_size):
                            iid_loss, _ = IID_loss(prob[:,i,:].reshape(batch_size, -1), prob[:,j,:].reshape(batch_size, -1))
                            loss += k_iic * iid_loss

                interm = []
                for k in range(group_size):
                    interm.append(net.encoder_common(inputs[:,k]))                    

                for c in range(net.num_cluster):
                    content_mu = []
                    content_sig = []
                    view_mu = []
                    view_sig = []
                    for k in range(group_size):
                        content_mu.append(net.encoder_content_mu[c](interm[k]))
                        content_sig.append(net.encoder_content_sig[c](interm[k]))
                        view_mu.append(net.encoder_view_mu[c](interm[k]))
                        view_sig.append(net.encoder_view_sig[c](interm[k]))
                    
                    content_mu = torch.stack(content_mu, dim=0)
                    content_prec = 1 / torch.stack(content_sig, dim=0) ** 2

                    content_mu = torch.sum(content_mu * content_prec, dim=0) / torch.sum(content_prec, dim=0)
                    content_sig = torch.sqrt(1 / torch.sum(content_prec, dim=0))
                    
                    kl_content = torch.sum(kldiv_gaussian(content_mu, content_sig), (1, 2, 3))
                    loss += -k_kldiv_latent * torch.sum(cluster_posterior[:, c] * kl_content, 0)
    
                    if torch.isnan(loss).any():
                        print('loss (kl) is nan (cluster={}, content)'.format(c))
                    
                    for k in range(group_size):
                        kl_view = torch.sum(kldiv_gaussian(view_mu[k], view_sig[k]), (1, 2, 3))
                        loss += -k_kldiv_latent * torch.sum(cluster_posterior[:, c] * kl_view, 0)
                        
                        if torch.isnan(loss).any():
                            print('loss (kl) is nan (cluster={}, view={})'.format(c, k))
    
                    content_latent = add_noise(content_mu, content_sig)
                    
                    for k in range(group_size):
                        view_latent = add_noise(view_mu[k], view_sig[k])                    
                        latent = torch.cat((content_latent, view_latent), dim=1)
                        recons = net.decoder_common(net.decoder[c](latent))
                        mse = 0.5 * torch.sum((recons - inputs[:,k]) ** 2 + 0.5 * np.log(2 * np.pi), (1, 2, 3))
                        loss += k_mse * torch.sum(cluster_posterior[:, c] * mse, 0)
    
                        if torch.isnan(loss).any():
                            print('loss (MSE) is nan (cluster={})'.format(c))
  
                
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
            net.info['train_k_mse'] = k_mse
            net.info['train_k_kldiv_cluster'] = k_kldiv_cluster
            net.info['train_k_kldiv_latent'] = k_kldiv_latent
            net.info['train_a_posterior'] = a_posterior
        
        print('Loss: {:.4f}'.format(epoch_loss))
        
    print('Training done')
    return epoch_loss



def kldiv_gaussian(mu, sig):
    return 0.5 * (1 + torch.log(sig ** 2 + eps) - mu ** 2 - sig ** 2)

def add_noise(mu, sig):
    noise = torch.randn_like(mu)
    return mu + sig * noise
    
    

    
    