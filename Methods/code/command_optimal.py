#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 19:29:23 2021

@author: fht
"""



import argparse
import os
import sys
import torch

from main_function import cross5CV
from main_function import indentTraTes


def getDefaultPara(opt):
    opt.nfold = 5   ### number of cross validation
    opt.prin = 1   ### print message or not
    
    opt.mid_dim = 256   ### the embedding size of MVGCN
    opt.num_layer = 3
    opt.alp = 0.5
    opt.beta = 0.5
    

    
    opt.NIPFusionType = 'add'   ### the fusion type of the NIP layey: choices=['add','cat']
    opt.normalizeType = 'row_sum_one'   ### normalize type of the initial feature
    opt.initFunType = 'kaiming_normal'   ### choices=['xavier_normal','xavier_uniform','kaiming_normal','kaiming_uniform']
    opt.actType = 'raw'   ### the activation function of the NIP layer: choices=['raw','sigmoid','tanh','relu','leaky_relu']
    opt.decoder_type = 'cat'   ### the decoder type of the model: choices=['none','ncf_linear','w','vec','cat']
    
    opt.sigmoid_flag = 1   ### sigmoid or not in the output layer of the model: choices=[0,1]
    opt.lossType = 'MF_all'   ### loss function type of the model: choices=['cross_entropy','MF_all','MSE']
    
    opt.lr = 0.0005   ### learning rate value of the model
    opt.weight_decay = 0.   ### the weight decay value of the model
    opt.dropProb = 0.1   ### the dropout probability value of the model
    opt.batch_size = 128   ### batch size of the model
    
    opt.epochs = 300   ### the max epoch value of the model: default 100
    opt.patience = 50   ### the patience value of the early stopping
    opt.earlyFlag = 1   ### early stopping or not: choices=[0,1]
    
    # opt.device = 'cpu'   ### the device of the running environment: choices=['cpu','cuda','cuda:0','cuda:1']
    
    
    opt.seed = 1   ### seed value of random functions
    opt.crossKey = 'cross'
    opt.indentKey = 'indent'
    
    opt.result_key = opt.exp_name   ### the key of the output file
    return

def optimal_para(dataName):
    mid_dim = 256   ### the embedding size of MVGCN
    num_layer = 3
    alp = 0.5
    beta = 0.5
    if dataName=='ZhangDDA':
        mid_dim = 256
        num_layer = 1
        alp = 0.1
        beta = 0.4

    return mid_dim, num_layer, alp, beta


def getArg():
    parser=argparse.ArgumentParser(description='MVGCN: data integration through multi-view graph convolutional '
                                   'network for predicting links in biomedical bipartite networks')
    
    parser.add_argument('--dataName', default='ZhangDDA', 
                            help='the data name')
    parser.add_argument('--exp_name', default='optimal_indent', 
                        choices=['mid_dim','num_layer', 'alp_beta', 
                                 'optimal_cross', 'optimal_indent'],
                        help='experiment name, just be optimal_ if no need for hyperparameter analysis,'
                             'and optimal_cross for cross validation experiment; optimal_indent for indenpendent experiment')
    parser.add_argument('--seed_cross', type=int)
    parser.add_argument('--seed_indent', type=int)
    parser.add_argument('--device', default= 'cpu', 
                        help='the device of the running environment, such as cpu.')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = getArg()
    getDefaultPara(opt)
    
    dataName = opt.dataName
    if not os.path.exists(dataName + '/'):
        os.mkdir(dataName + '/')
    
    opt.mid_dim, opt.num_layer, opt.alp, opt.beta = optimal_para(dataName)
    
    seed = opt.seed
    # use cuda or not
    if opt.device.startswith('cuda') & (torch.cuda.is_available()):
        opt.cudaFlag = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        opt.cudaFlag = False
        opt.device = 'cpu'
    
    exp_name = opt.exp_name
    
            
    # corss validation for the hyperparameter "embedding size"
    if exp_name=='mid_dim':
        mid_dim_ls = [8,16,32,64,128,256]
        for mid_dim in mid_dim_ls:
            opt.mid_dim = mid_dim
            cross5CV(opt)
    # cross validation for the hyperparameter "number of layers" all layer for (default) attention
    elif exp_name=='num_layer':
        num_layer_ls = [1,2,3,4,5]
        for num_layer in num_layer_ls:
            opt.num_layer = num_layer
            cross5CV(opt)
    # cross validation for the hyperparameter "alpha" and "beta"
    elif exp_name=='alp_beta':
        alp_ls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        beta_ls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for alp in alp_ls:
            for beta in beta_ls:
                print('alpha: {:.1f}, beta: {:.1f}'.format(alp, beta))
                opt.alp, opt.beta = alp, beta
                cross5CV(opt)
    
    # the optimal hyperparameters for modeling
    elif exp_name.startswith('optimal'):
        if exp_name.endswith('cross'):
            cross5CV(opt)
        elif exp_name.endswith('indent'):
            indentTraTes(opt)
    
    else:
        print('experiment is wrong!')
        sys.exit(1)
        
        
        