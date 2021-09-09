#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 02:32:28 2021

@author: FHt
"""


import copy
import numpy as np
import scipy.sparse as sp
import sys
import time
import torch
import torch.optim as optim

from sklearn.metrics import roc_auc_score, average_precision_score

from model_MF import MF
from process_feature import getFeature
from utils.torch_data import getLoader

from utils.loss_function import lossF
from utils.clac_metric import get_metrics
from utils.normalization import normalizeRow, normalizeRowCol
from utils.process_set import diagZeroAdj,get_profile_sim

def getBigAdj(sim_A_total, sim_b_total, traiY):
    adj_net_homo = np.array([np.vstack((np.hstack((sim_A_total[i], np.zeros_like(traiY))),
                                    np.hstack((np.zeros_like(traiY.T), sim_b_total[j]))
                          )) for i in range(len(sim_A_total)) for j in range(len(sim_b_total))]
                        )
    adj_net_hete = np.array([np.vstack((np.hstack((np.zeros_like(sim_A_total[i]), traiY)),
                               np.hstack((traiY.T, np.zeros_like(sim_b_total[j])))
                              )) for i in range(len(sim_A_total)) for j in range(len(sim_b_total))]
                            )
    
    for i in range(len(adj_net_homo)):
        adj_net_homo[i] = normalizeRowCol(adj_net_homo[i])
        adj_net_hete[i] = normalizeRowCol(adj_net_hete[i])    
    return torch.FloatTensor(adj_net_homo), torch.FloatTensor(adj_net_hete)

def testModel(opt, model, loader, F_u, F_i, adj_net_homo, adj_net_hete):
    cudaFlag = opt.cudaFlag; device = opt.device
    prin = opt.prin
    lossType = opt.lossType
    
    model.eval()
    lossLs, y_label, y_pred = [], [], []
    with torch.no_grad():
        for i, (idx0, idx1, y) in enumerate(loader):
            if cudaFlag:
                y = y.to(device)
            output = model(opt, F_u, F_i, idx0, idx1, adj_net_homo, adj_net_hete)
            loss = lossF(lossType, output, y).item()
            lossLs.append(loss)
            
            y_label += y.cpu().numpy().tolist()
            y_pred += output.detach().cpu().numpy().tolist()
        if prin:
            aveLoss = sum(lossLs) / float(len(lossLs))
            print('average test loss: {:.4f}'.format(aveLoss))
    aupr = average_precision_score(y_label, y_pred)
    auc = roc_auc_score(y_label, y_pred)
    return y_label, y_pred, aupr, auc, loss

def trainModel(opt, model, optimizer, adj_net_homo, adj_net_hete, F_u, F_i, trai_loader, vali_loader, test_loader, tes_list):
    epochs = opt.epochs
    lossType = opt.lossType
    earlyFlag = opt.earlyFlag
    patience = opt.patience
    device = opt.device
    prin = opt.prin

    adj_net_homo = adj_net_homo.to(device)
    adj_net_hete = adj_net_hete.to(device)
    F_u = F_u.to(device)
    F_i = F_i.to(device)
    
    if prin:
        print('Start Training...')
    # Train model
    t_total = time.time() 
    # initialization
    model_max = copy.deepcopy(model)
    max_auc = 0
    loss_history = []
    epoch_patience = 0
    epoch = 0
    epoch_best = 0
    try:
        for epoch in range(epochs):
            epoch_start = time.time()
            if prin:
                print('\n-------- Epoch {:04d} --------'.format(epoch))
            y_pred_train = []
            y_label_train = []            
            # train epoch
            model.train()
            for i, (idx0, idx1, y) in enumerate(trai_loader):
                idx0 = idx0.to(device)
                idx1 = idx1.to(device)
                y = y.to(device)
                    
                optimizer.zero_grad()
                output = model(opt, F_u, F_i, idx0, idx1, adj_net_homo, adj_net_hete)
                loss_train = lossF(lossType, output, y)
                loss_train.backward()                
                optimizer.step()
                loss_history.append(loss_train.detach().cpu().item())
                y_label_train += y.cpu().numpy().tolist()
                y_pred_train += output.detach().cpu().numpy().tolist()
                if prin:
                    if i == 0:
                        print('epoch: {:04d}'.format(epoch),
                              '/ train iteration: {:04d}'.format(i),
                              '/ train loss: {:.4f}'.format(loss_train.item()))
                        
            aupr_train = average_precision_score(y_label_train, y_pred_train)
            auc_train = roc_auc_score(y_label_train, y_pred_train)
            
            epoch_patience += 1
            if epoch_patience > patience:
                print('best epoch is: {:04d} \n early stopping at epoch: {:04d}'.format(epoch_best, epoch))
                break
            # validation after each epoch
            if earlyFlag:
                model.eval()
                y_label_val, y_pred_val, aupr_val, auc_val, loss_val = testModel(opt, model, vali_loader, F_u, F_i, adj_net_homo, adj_net_hete)
                if auc_val > max_auc:
                    epoch_best = epoch
                    epoch_patience = 0
                    model_max = copy.deepcopy(model)
                    # y_label_opt, y_pred_opt, aupr_opt, auc_opt, loss_opt = y_label_val, y_pred_val, aupr_val, roc_val, loss_val
                    # torch.save(model, 'optimal_model.pkl')
                    max_auc = auc_val                    
                    if prin:
                        print(' epoch:       {:04d}\n'.format(epoch),
                              ' loss_train:  {:.4f}\n'.format(loss_train.item()),
                              ' aupr_train:  {:.4f}\n'.format(aupr_train),
                              ' auroc_train: {:.4f}\n'.format(auc_train),
                              ' loss_val:    {:.4f}\n'.format(loss_val),
                              ' aupr_val:    {:.4f}\n'.format(aupr_val),
                              ' auc_val:     {:.4f}\n'.format(auc_val),
                              ' max_auc:     {:.4f}'.format(max_auc))
            else:
                model_max = copy.deepcopy(model)
            if prin:
                print('the {:04d} epoch take {:.4f} seconds'.format(epoch, time.time()-epoch_start))
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        pass
    # plt.plot(loss_history)
    if prin:
        print("Optimization Finished!")
        print("Total time elapsed: {:.2f}s".format(time.time() - t_total))
        print('The best epoch is: {:04d} \n current epoch is: {:04d}'.format(epoch_best, epoch))
    # Testing
    model_max.eval()
    test_label, test_score, aupr_test, auc_test, loss_test = testModel(opt, model_max, test_loader, F_u, F_i, adj_net_homo, adj_net_hete)
    # test_label, test_score, aupr_test, auc_test, loss_test = y_label_opt, y_pred_opt, aupr_opt, auc_opt, loss_opt
    if prin:
        print('loss_test: {:.4f}\n'.format(loss_test),
              'aupr_test: {:.4f}\n'.format(aupr_test), 
              ' auc_test: {:.4f}'.format(auc_test))

    return test_label, test_score, model_max

def trainTestMain(opt, sim_A, sim_b, tra_list, val_list, tes_list):
    opt.att_dim = opt.mid_dim
    batch_size = opt.batch_size
    lr = opt.lr
    weight_decay = opt.weight_decay
    device = opt.device

    
    opt.row_num = row_num = sim_A[0].shape[0]
    opt.col_num = col_num = sim_b[0].shape[0]
    
    trai_loader = getLoader(batch_size, tra_list)
    vali_loader = getLoader(batch_size, val_list)
    test_loader = getLoader(batch_size, tes_list)
    
    sim_A_total, sim_b_total = copy.deepcopy(sim_A), copy.deepcopy(sim_b)
    sim_A_total = get_profile_sim(sim_A_total, tra_list, row_num, col_num)
    tra_T = np.zeros_like(tra_list)
    tra_T[:,0] = tra_list[:,1]
    tra_T[:,1] = tra_list[:,0]
    tra_T[:,2] = tra_list[:,2]
    sim_b_total = get_profile_sim(sim_b_total, tra_T, col_num, row_num)
    
    sim_A_total = diagZeroAdj(sim_A_total)
    sim_b_total = diagZeroAdj(sim_b_total)
    

        
    traiY = sp.coo_matrix((tra_list[:,2], (tra_list[:,0],tra_list[:,1])),
                          shape=(row_num, col_num), dtype=np.float32).toarray()
    
    adj_net_homo, adj_net_hete = getBigAdj(sim_A_total, sim_b_total, traiY)
    
    F_u, F_i = getFeature(opt, traiY, sim_A_total, sim_b_total)
    F_u, F_i = torch.FloatTensor(F_u), torch.FloatTensor(F_i)
    
    opt.num_sim = num_sim = len(sim_A_total) * len(sim_b_total)
    model = MF(opt, num_sim).to(device)
    optimizer = optim.Adam(model.parameters(),
                   lr=lr, weight_decay=weight_decay)
    test_label, test_score, model_max = trainModel(opt, model, optimizer, adj_net_homo, adj_net_hete, F_u, F_i, trai_loader, vali_loader, test_loader, tes_list)

    criteria_result = get_metrics(np.mat(test_label), np.mat(test_score))
    return test_label, test_score, criteria_result, model_max, F_u, F_i

