#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:12:44 2020

A python implementation of CIGMO (Categorical Invariant Generative MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""

from tqdm import tqdm
import numpy as np
import numpy.random
from types import SimpleNamespace

from scipy.optimize import linear_sum_assignment
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import adjusted_rand_score

import utils

#%%

def test_model(net, dataloader, class_dict=None, device='cpu', disable_tdqm=False):

    net.to(device)
    net.eval()

    classes = []
    # subclasses = []
    clusters = []
    cposts = []
    recons = []
    inputs = []
    contents = []
    views = []
    
    with torch.set_grad_enabled(False):    
        for inps, labs in tqdm(dataloader, position=0, leave=True, disable=disable_tdqm):
            recs = torch.zeros_like(inps)     
            cs = torch.zeros(inps.size(0), net.content_dim, 1, 1)
            vs = torch.zeros(inps.size(0), net.view_dim, 1, 1)
            inps = inps.to(device)
    
            cpost = torch.softmax(net.classifier(inps).reshape(-1, net.num_cluster), 1)
            clu = torch.argmax(cpost, 1).reshape(-1)
            
            for c in range(net.num_cluster):
                idx = (clu == c).nonzero(as_tuple=True)
                if idx[0].nelement() == 0: continue
                interm = net.encoder_common(inps[idx])
                content = net.encoder_content_mu[c](interm)
                view = net.encoder_view_mu[c](interm)
                rec = net.decoder_common(net.decoder[c](torch.cat((content, view), 1)))
                recs[idx] = rec.detach().cpu()
                cs[idx] = content.detach().cpu()
                vs[idx] = view.detach().cpu()
    
            recons.append(recs)
            clusters.append(clu.detach().cpu())
            cposts.append(cpost.detach().cpu())
            lab0 = labs[0] if type(labs)==list else labs
            if class_dict == None:
                classes.append(lab0)
            else:
                classes.append(torch.tensor([class_dict[lab] for lab in lab0]))
            # subclasses.append(labs[1])
            inputs.append(inps.detach().cpu())
            contents.append(cs)
            views.append(vs)

    res = SimpleNamespace()
    res.classes = torch.cat(classes, 0)
    # res.subclasses = torch.cat(subclasses, 0)
    res.clusters = torch.cat(clusters, 0)
    res.cposts = torch.cat(cposts, 0)
    res.recons = torch.cat(recons, 0)
    res.inputs = torch.cat(inputs, 0)
    res.contents = torch.cat(contents, 0)
    res.views = torch.cat(views, 0)
    
    return res


def swapping(net, c, list1, list2, column_content=True, device='cpu'):
    
    net.to(device)
    net.eval()
    
    content = net.encoder_content_mu[c](net.encoder_common(list1))
    view = net.encoder_view_mu[c](net.encoder_common(list2))
    images = []
    disps = [torch.zeros_like(list1[0])]
    len1 = list1.size(0)
    len2 = list2.size(0)
    for i in range(len1): disps.append(list1[i])    
    for j in range(len2):
        disps.append(list2[j])
        images = []
        for i in range(len1):
            img = net.decoder_common(net.decoder[c](torch.cat((content[i:i+1], view[j:j+1]), dim=1)))
            images.append(img[0])
            disps.append(img[0])
            
    disps = torch.stack(disps, dim=0)
    
    nrow = len1+1
    
    if not column_content:
        disps = disps.reshape(len2+1, len1+1, disps.size(1), disps.size(2), disps.size(3))
        disps = disps.permute(1, 0, 2, 3, 4)
        disps = disps.reshape((len2+1)*(len1+1), disps.size(2), disps.size(3), disps.size(4))
        nrow = len2+1
    
    disp = torchvision.utils.make_grid(disps, nrow=nrow, padding=1, pad_value=0, normalize=True)
    disp[:, img[0].size(1)-1, :] = 1    
    disp[:, :, img[0].size(2)-1] = 1    
    return torch.cat(images, dim=0), disp

def swapping2(net, list1, list2, device='cpu'):
    
    net.to(device)
    net.eval()
    
    images = []
    disps = [torch.zeros_like(list1[0])]
    for i in range(list1.size(0)): disps.append(list1[i])    
    for j in range(list2.size(0)):
        disps.append(list2[j])
        for i in range(list1.size(0)):
            cpost = torch.softmax(net.classifier(list1[i:i+1]).reshape(net.num_cluster), 0)
            clu = torch.argmax(cpost)
            content = net.encoder_content_mu[clu](net.encoder_common(list1[i:i+1]))
            view = net.encoder_view_mu[clu](net.encoder_common(list2[j:j+1]))
            img = net.decoder_common(net.decoder[clu](torch.cat((content, view), dim=1)))
            images.append(img[0])
            disps.append(img[0])

    disps = torch.stack(disps, dim=0)
    disp = torchvision.utils.make_grid(disps, nrow=list1.size(0)+1, padding=1, pad_value=0, normalize=True)
    disp[:, img[0].size(1)-1, :] = 1    
    disp[:, :, img[0].size(2)-1] = 1    
    return torch.cat(images, dim=0), disp


def purge_large_data(res):
    res.inputs = []
    res.recons = []
    res.contents = []
    res.views = []
    
def calc_outputs(net, dataloader, class_dict=None, device='cpu', disable_tdqm=False):

    net.to(device)
    net.eval()

    classes = []
    clusters = []
    cposts = []
    contents = []
    views = []
    
    with torch.set_grad_enabled(False):    
        for inps, labs in tqdm(dataloader, position=0, leave=True, disable=disable_tdqm):
            cs = torch.zeros(inps.size(0), net.content_dim, 1, 1)
            vs = torch.zeros(inps.size(0), net.view_dim, 1, 1)
            inps = inps.to(device)
    
            cpost = torch.softmax(net.classifier(inps).reshape(-1, net.num_cluster), 1)
            clu = torch.argmax(cpost, 1).reshape(-1)
            
            for c in range(net.num_cluster):
                idx = (clu == c).nonzero(as_tuple=True)
                if idx[0].nelement() == 0: continue
                interm = net.encoder_common(inps[idx])
                content = net.encoder_content_mu[c](interm)
                view = net.encoder_view_mu[c](interm)
                cs[idx] = content.detach().cpu()
                vs[idx] = view.detach().cpu()
    
            clusters.append(clu.detach().cpu())
            cposts.append(cpost.detach().cpu())
            lab0 = labs[0] if type(labs)==list else labs
            if class_dict == None:
                classes.append(lab0)
            else:
                classes.append(torch.tensor([class_dict[lab] for lab in lab0]))
            contents.append(cs)
            views.append(vs)

    res = SimpleNamespace()
    res.classes = torch.cat(classes, 0)
    res.clusters = torch.cat(clusters, 0)
    res.cposts = torch.cat(cposts, 0)
    res.contents = torch.cat(contents, 0)
    res.views = torch.cat(views, 0)
    
    return res

def calc_clustering(net, dataloader, class_dict=None, device='cpu', disable_tdqm=False, keep_inputs=False):

    net.to(device)
    net.eval()

    classes = []
    clusters = []
    cposts = []
    inputs = []    
    
    with torch.set_grad_enabled(False):    
        for inps, labs in tqdm(dataloader, position=0, leave=True, disable=disable_tdqm):
            inps = inps.to(device)
    
            cpost = torch.softmax(net.classifier(inps).reshape(-1, net.num_cluster), 1)
            clu = torch.argmax(cpost, 1).reshape(-1)
            
            clusters.append(clu.detach().cpu())
            cposts.append(cpost.detach().cpu())
            lab0 = labs[0] if type(labs)==list else labs
            if class_dict == None:
                classes.append(lab0)
            else:
                classes.append(torch.tensor([class_dict[lab] for lab in lab0]))
            inputs.append(inps.detach().cpu())

    res = SimpleNamespace()
    res.classes = torch.cat(classes, 0)
    res.clusters = torch.cat(clusters, 0)
    res.cposts = torch.cat(cposts, 0)
    if keep_inputs: res.inputs = torch.cat(inputs, 0)
    
    return res

def calc_embedding(net, dataloader, class_dict=None, device='cpu', disable_tdqm=False):

    net.to(device)
    net.eval()

    embeds = []
    
    with torch.set_grad_enabled(False):    
        for inps, labs in tqdm(dataloader, position=0, leave=True, disable=disable_tdqm):
            es = torch.zeros(inps.size(0), net.proto_dim, 1, 1)
            inps = inps.to(device)    
            embed = net.encoder(inps)
            embeds.append(embed.detach().cpu())

    res = SimpleNamespace()
    res.embeds = torch.cat(embeds, 0)
    
    return res


def gen_random_images(net, cluster, num, device='cpu', content_std=None, view_std=None, rs=None):

    net.to(device)
    net.eval()

    if rs is None:
        c = torch.randn(num, net.content_dim, 1, 1)
        v = torch.randn(num, net.view_dim, 1, 1)
    else:
        c = torch.tensor(rs.normal(size=(num, net.content_dim, 1, 1)), dtype=torch.float32)
        v = torch.tensor(rs.normal(size=(num, net.view_dim, 1, 1)), dtype=torch.float32)
    if content_std is not None:
        c = c * content_std.reshape(1, net.content_dim, 1, 1)
    if view_std is not None:
        v = v * view_std.reshape(1, net.view_dim, 1, 1)
    r = net.decoder_common(net.decoder[cluster](torch.cat((c, v), dim=1)))

    return r.cpu().detach()

def gen_random_images2(net, cluster, list1_size, list2, device='cpu'):
        
    net.to(device)
    net.eval()

    rs = []
    for i in range(len(list2)):
        v = net.encoder_view_mu[cluster](net.encoder_common(list2[i:i+1])).repeat(list1_size, 1, 1, 1)
        c = torch.randn(list1_size, net.content_dim, 1, 1)
        r = net.decoder_common(net.decoder[cluster](torch.cat((c, v), dim=1)))
        rs.append(r)
        
    return torch.cat(rs, dim=0).cpu().detach()

def gen_2d_images(net, cluster, dim1, dim2, pts, view_input, device='cpu'):
        
    net.to(device)
    net.eval()

    v = net.encoder_view_mu[cluster](net.encoder_common(view_input)).repeat(len(pts)**2, 1, 1, 1)

    c1, c2 = torch.meshgrid(pts, pts)
    c1 = c1.reshape(-1)
    c2 = c2.reshape(-1)
    c = torch.zeros(c1.size(0), net.content_dim, 1, 1)
    c[:, dim1, 0, 0] = c1
    c[:, dim2, 0, 0] = c2
    r = net.decoder_common(net.decoder[cluster](torch.cat((c, v), dim=1)))

    return r.cpu().detach()

def gen_2d_images2(net, cluster, input1, input2, grid_size, device='cpu'):
        
    net.to(device)
    net.eval()

    v1 = net.encoder_view_mu[cluster](net.encoder_common(input1)).repeat(grid_size**2, 1, 1, 1)
    v2 = net.encoder_view_mu[cluster](net.encoder_common(input2)).repeat(grid_size**2, 1, 1, 1)
    c1 = net.encoder_content_mu[cluster](net.encoder_common(input1)).repeat(grid_size**2, 1, 1, 1)
    c2 = net.encoder_content_mu[cluster](net.encoder_common(input2)).repeat(grid_size**2, 1, 1, 1)

    u1, u2 = torch.meshgrid(torch.linspace(0, 1, grid_size), torch.linspace(0, 1, grid_size))
    u1 = u1.reshape(-1, 1, 1, 1)
    u2 = u2.reshape(-1, 1, 1, 1)
    v = u1 * v1 + (1 - u1) * v2
    c = u2 * c1 + (1 - u2) * c2
    r = net.decoder_common(net.decoder[cluster](torch.cat((c, v), dim=1)))

    return r.cpu().detach()

def clustering_accuracy(clusters, num_cluster, score, labels):

    est_labels = np.zeros(len(labels))
    cluster_to_label = np.zeros(num_cluster)-1
    
    for k in range(num_cluster):
        idx = (clusters == k).nonzero()[0]
        if len(idx) == 0: continue
        max_idx = idx[score[idx, k].argmax()]
        cluster_to_label[k] = labels[max_idx]
        est_labels[idx] = labels[max_idx]
    
    acc = (est_labels == labels).sum() / len(labels)
    return acc, est_labels, cluster_to_label
    
def best_assignment_accuracy(clusters, num_cluster, labels, all_labels):
    # This works only when num_cluster == num_label

    num_label = len(all_labels)    
    cost_matrix = np.zeros((num_cluster, num_label))
    for k in range(num_cluster):
        for l in range(num_label):
            cost_matrix[k, l] = -(labels[clusters == k] == all_labels[l]).sum()
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    est_labels = np.zeros_like(labels)
    for k in range(num_cluster):
        idx = (row_ind == k).nonzero()[0]
        if len(idx) == 0:  continue
        est_labels[clusters == k] = all_labels[col_ind[idx[0]]]
    acc = (est_labels == labels).sum() / len(labels)
    
    return acc, est_labels

def ari(clusters, labels):
    return adjusted_rand_score(labels, clusters)

def reconstrution_accuracy(inputs, recons):
    
    sqerr = torch.sum((inputs - recons) ** 2, (1,2,3))
    sqnorm = torch.sum(inputs ** 2, (1,2,3))
    
    return sqerr / sqnorm

def swapping_accuracy(net, res, n_view, device='cpu', n_sample=1000):
    
    net.to(device)
    net.eval()    
    rs = numpy.random.RandomState(seed=0)

    cidx = rs.permutation(res.inputs.size(0))[:n_sample]
    contents = res.contents[cidx].to(device)
    clusters = res.clusters[cidx].to(device)
    vidx = rs.permutation(res.inputs.size(0))[:n_sample]
    views = res.views[vidx].to(device)
    ridx = np.floor(cidx / n_view) * n_view + (vidx % n_view)
    inputs = res.inputs[ridx]

    with torch.set_grad_enabled(False):    
    
        res.swap_gen = torch.zeros(n_sample, res.inputs.size(1), res.inputs.size(2), res.inputs.size(3))
        for i in range(n_sample):
            c = clusters[i]
            lat = torch.cat((contents[i:i+1], views[i:i+1]), dim=1)
            res.swap_gen[i] = net.decoder_common(net.decoder[c](lat)).cpu()
    
        swap_acc = reconstrution_accuracy(inputs, res.swap_gen)

    return swap_acc        
    
def swapping_accuracy2(net, res, n_view, device='cpu', n_sample=1000):
    
    net.to(device)
    net.eval()    
    rs = numpy.random.RandomState(seed=0)

    sample_idx = torch.tensor(rs.permutation(res.inputs.size(0))[:n_sample])
    clusters = res.clusters[sample_idx]

    with torch.set_grad_enabled(False):    
    
        inputs = torch.zeros(n_sample, res.inputs.size(1), res.inputs.size(2), res.inputs.size(3))
        res.swap_gen = torch.zeros(n_sample, res.inputs.size(1), res.inputs.size(2), res.inputs.size(3))
        for c in range(net.num_cluster):
            sidx = (clusters == c).nonzero().squeeze()
            if sidx.nelement() == 0:             
                continue
            cidx = sample_idx[sidx]
            sample_idx2 = (res.clusters == c).nonzero().squeeze()
            vidx = sample_idx2[rs.permutation(sample_idx2.nelement())[:cidx.nelement()]]
            ridx = np.floor_divide(cidx, n_view) * n_view + (vidx % n_view)
            contents = res.contents[cidx].to(device) if cidx.nelement()>1 else res.contents[cidx:cidx+1].to(device)
            views = res.views[vidx].to(device) if vidx.nelement()>1 else res.views[vidx:vidx+1].to(device)
            inputs[sidx] = res.inputs[ridx]
            lat = torch.cat((contents, views), dim=1)
            res.swap_gen[sidx] = net.decoder_common(net.decoder[c](lat)).cpu()
    
        swap_acc = reconstrution_accuracy(inputs, res.swap_gen)

    return swap_acc  

def fewshot_accuracy(latents, n_view, n_shot=1, n_repetition=20, metric='cosine', device='cpu'):
    
    accs = []
    rs = numpy.random.RandomState(seed=0)

    n_shape = int(latents.size(0) / n_view)
    latent_dim = latents.size(1)

    for j in range(n_repetition):
        lat = latents.reshape(n_shape, n_view, latent_dim)
        gallery = torch.zeros(n_shape, n_shot, latent_dim)
        probe = torch.zeros(n_shape, n_view - n_shot, latent_dim)
        
        for i in range(n_shape):
            idx = rs.permutation(n_view)
            gallery[i, :, :] = lat[i, idx[:n_shot], :]
            probe[i, :, :] = lat[i, idx[n_shot:], :]
            
        gallery = gallery.reshape(-1, latent_dim).to(device)
        probe = probe.reshape(-1, latent_dim).to(device)
        if metric == 'cosine':
            probe = probe.permute(1, 0)
            dist = 1 - torch.mm(gallery / torch.norm(gallery, dim=1, keepdim=True), probe / torch.norm(probe, dim=0, keepdim=True))  
        elif metric == 'euclidean':
            dist = torch.cdist(gallery, probe)
        else:
            print('not supported metric:', metric)
            return None
        minidx = torch.argmin(dist, dim=0).cpu()
        est_shape = torch.floor(minidx / float(n_shot))
        true_shape = torch.floor(torch.arange(0, n_shape * (n_view - n_shot)) / float(n_view - n_shot))
        acc = float(torch.sum(est_shape == true_shape)) / float(len(true_shape))
        accs.append(acc)

    chance_acc = 1 / float(n_shape)

    return accs, chance_acc

#%%

def fewshot_accuracy_ids(latents, shape_ids, n_shot=1, n_repetition=20, metric='cosine', device='cpu'):
    
    accs = []
    rs = numpy.random.RandomState(seed=0)

    n_sample = latents.size(0)
    all_shape_ids = np.unique(shape_ids)
    n_shape = len(all_shape_ids)
    latent_dim = latents.size(1)
    
    ids_count = {}
    ids_dict = {}
    
    for sid in all_shape_ids:
        ids_count[sid] = np.sum(shape_ids == sid)
        ids_dict[sid] = np.where(shape_ids == sid)[0]
    n_view = max(ids_count.values())
    
    for j in range(n_repetition):
        gallery = torch.zeros(n_shape, n_shot, latent_dim)
        probe = torch.zeros(n_shape, n_view - n_shot, latent_dim)
        valid = torch.zeros(n_shape, n_view - n_shot, dtype=torch.bool)
        gallery[:] = np.NaN
        probe[:] = np.NaN
        
        for i in range(n_shape):
            idx = ids_dict[all_shape_ids[i]]
            idx = rs.permutation(idx)
            gallery[i, :, :] = latents[idx[:n_shot], :]
            probe[i, :len(idx)-n_shot, :] = latents[idx[n_shot:], :]
            valid[i, :len(idx)-n_shot] = True
            
        gallery = gallery.reshape(-1, latent_dim).to(device)
        probe = probe.reshape(-1, latent_dim).to(device)
        valid = valid.reshape(-1)
        if metric == 'cosine':
            probe = probe.permute(1, 0)
            dist = 1 - torch.mm(gallery / torch.norm(gallery, dim=1, keepdim=True), probe / torch.norm(probe, dim=0, keepdim=True))  
        elif metric == 'euclidean':
            dist = torch.cdist(gallery, probe)
        else:
            print('not supported metric:', metric)
            return None
        minidx = torch.argmin(dist, dim=0).cpu()
        est_shape = torch.floor(minidx / float(n_shot))
        true_shape = torch.floor(torch.arange(0, n_shape * (n_view - n_shot)) / float(n_view - n_shot))
        acc = float(torch.sum(est_shape[valid] == true_shape[valid])) / float(sum(valid))
        accs.append(acc)

    chance_acc = 1 / float(n_shape)

    return accs, chance_acc

#%%
    
def classify1(train_in, train_tg, test_in, test_tg, n_class, device='cpu'):
    n_interm = 500
    n_epochs = 50
    batch_size = 2000
    mlp = nn.Sequential(
        nn.Linear(train_in.size(1), n_interm),
        nn.ReLU(inplace=True),
        nn.Linear(n_interm, n_class)).to(device)
    train_dataset = data.TensorDataset(train_in.to(device), train_tg.to(device))
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = data.TensorDataset(test_in.to(device), test_tg.to(device))
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(params=utils.get_params(mlp), lr=0.001)
    lossfun = nn.CrossEntropyLoss()
    
    with torch.set_grad_enabled(True):
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for inputs, targets in train_dataloader:
                optimizer.zero_grad()
                outputs = mlp(inputs)
                loss = lossfun(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu()
            # print('Epoch {}/{}: loss = {:.4f}'.format(epoch + 1, n_epochs, epoch_loss))
    
    nhit = 0
    for inputs, targets in test_dataloader:
        outputs = mlp(inputs)
        nhit += (outputs.argmax(dim=1) == targets).detach().cpu().sum().item()
        
    return float(nhit) / test_in.size(0)
        
def classify_cv(inputs, targets, n_class, n_split, seed=0, device='cpu'):
    rs = numpy.random.RandomState(seed=seed)
    idx = rs.permutation(inputs.size(0))
    inputs = inputs[idx]
    targets = targets[idx]
    sp = torch.linspace(0, inputs.size(0), n_split+1).int()
    rates = []
    for i in range(n_split):
        print('split #{}/{}'.format(i, n_split))
        train_inputs = torch.cat([inputs[:sp[i]], inputs[sp[i+1]:]], dim=0)
        train_targets = torch.cat([targets[:sp[i]], targets[sp[i+1]:]], dim=0)
        test_inputs = inputs[sp[i]:sp[i+1]]
        test_targets = targets[sp[i]:sp[i+1]]
        r = classify1(train_inputs, train_targets, test_inputs, test_targets, n_class, device=device)
        rates.append(r)
        print('--> {}'.format(r))
    return numpy.mean(rates)        

def classify_once(inputs, targets, n_class, split_ratio, seed=0, device='cpu'):
    rs = numpy.random.RandomState(seed=seed)
    idx = rs.permutation(inputs.size(0))
    inputs = inputs[idx]
    targets = targets[idx]
    sp = int(inputs.size(0) * split_ratio)
    train_inputs = inputs[:sp]
    train_targets = targets[:sp]
    test_inputs = inputs[sp:]
    test_targets = targets[sp:]
    r = classify1(train_inputs, train_targets, test_inputs, test_targets, n_class, device=device)
    return r

# def separability(clusters, shapes, views, n_cluster, n_view, device='cpu'):
#     datasize = shapes.size(0)
#     n_shape = int(datasize / n_view)
#     shape_ids = (torch.arange(0, datasize) / n_view).long()
#     view_ids = (torch.arange(0, datasize) % n_view).long()
#     rss = []
#     rsv = []
#     rvs = []
#     rvv = []
#     for c in range(n_cluster):
#         idx = (clusters == c).nonzero().squeeze()
#         if len(idx) == 0: continue
#         print('cluster #{}/{}, shape -> shape'.format(c,n_cluster))        
#         rss.append(classify_cv(shapes[idx], shape_ids[idx], n_shape, n_split=5, device=device))
#         print('cluster #{}/{}, shape -> view'.format(c,n_cluster))        
#         rsv.append(classify_cv(shapes[idx], view_ids[idx], n_view, n_split=5, device=device))
#         print('cluster #{}/{}, view -> shape'.format(c,n_cluster))        
#         rvs.append(classify_cv(views[idx], shape_ids[idx], n_shape, n_split=5, device=device))
#         print('cluster #{}/{}, view -> view'.format(c,n_cluster))        
#         rvv.append(classify_cv(views[idx], view_ids[idx], n_view, n_split=5, device=device))
#     return rss, rsv, rvs, rvv

def shape_view_separability(clusters, shapes, views, n_cluster, n_view, device='cpu'):
    datasize = shapes.size(0)
    n_shape = int(datasize / n_view)
    shape_ids = (torch.arange(0, datasize) / n_view).long()
    rss = []
    rvs = []
    rch = []
    for c in range(n_cluster):
        idx = (clusters == c).nonzero().squeeze()
        # print(idx.nelement())
        if idx.nelement() < 2: 
            rss.append(0)
            rvs.append(0)
            rch.append(0)
        else:
            print('cluster #{}/{}, shape -> shape'.format(c,n_cluster))        
            rss.append(classify_once(shapes[idx], shape_ids[idx], n_shape, split_ratio=0.8, device=device))
            print('cluster #{}/{}, view -> shape'.format(c,n_cluster))        
            rvs.append(classify_once(views[idx], shape_ids[idx], n_shape, split_ratio=0.8, device=device))
            rch.append(1 / len(shape_ids[idx].unique()))
    return rss, rvs, rch

#%%

def shape_view_separability_ids(clusters, shapes, views, n_cluster, shape_ids, device='cpu'):
    datasize = shapes.size(0)
    all_shape_ids = np.unique(shape_ids)
    n_shape = len(all_shape_ids)
    shape_dict = { all_shape_ids[i] : i for i in range(n_shape) }
    shape_ids2 = torch.tensor([shape_dict[sid] for sid in shape_ids])
    rss = []
    rvs = []
    rch = []
    for c in range(n_cluster):
        idx = (clusters == c).nonzero().squeeze()
        # print(idx.nelement())
        if idx.nelement() < 2: 
            rss.append(0)
            rvs.append(0)
            rch.append(0)
        else:
            print('cluster #{}/{}, shape -> shape'.format(c,n_cluster))        
            rss.append(classify_once(shapes[idx], shape_ids2[idx], n_shape, split_ratio=0.8, device=device))
            print('cluster #{}/{}, view -> shape'.format(c,n_cluster))        
            rvs.append(classify_once(views[idx], shape_ids2[idx], n_shape, split_ratio=0.8, device=device))
            rch.append(1 / len(shape_ids2[idx].unique()))
    return rss, rvs, rch

#%%

def construct_combine_latent(net, res, num_cluster, group_size):
    if num_cluster == 1:
        if group_size == 1:
            lat = torch.cat([res.contents, res.views], 1).squeeze()
        else:
            lat = res.contents.squeeze()        
    else:
        if group_size == 1:
            lat = torch.zeros(res.inputs.size(0), net.num_cluster, net.content_dim + net.view_dim)
            for j in range(res.inputs.size(0)):
                lat[j, res.clusters[j], :] = torch.cat([res.contents[j], res.views[j]], 0).squeeze()
        else:
            lat = torch.zeros(res.inputs.size(0), net.num_cluster, net.content_dim)
            for j in range(res.inputs.size(0)):
                lat[j, res.clusters[j], :] = res.contents[j].squeeze()
        lat = lat.reshape(res.inputs.size(0), -1)
    return lat

#%%

def rss_weighted_average(res):
    ns, _ = np.histogram(res.clusters, bins=np.arange(0, res.cposts.size(1)+1))
    # ns = ns[ns > 1]
    return np.sum(np.array(res.acc_shape_shape) * ns) / np.sum(ns)
    
def rvs_weighted_average(res):
    ns, _ = np.histogram(res.clusters, bins=np.arange(0, res.cposts.size(1)+1))
    # ns = ns[ns > 1]
    return np.sum(np.array(res.acc_view_shape) * ns) / np.sum(ns)

#%%

def num_effective_clusters(res):
    c = 0
    clu = res.clusters
    for i in range(clu.max()+1):
        if (clu == i).sum() > 0:
            c += 1

    return c

