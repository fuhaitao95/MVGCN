# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:05:11 2020
modified on Tue Jan 5 12点40分 2021
modified on Tue Jan 11 22点38分 2021
modified on Tue Jan 13 15点41分 2021
modified on Sat Feb 13 23点10分 2021
@author: xinxi
"""


import argparse
import os
import sys



def optPara():
    
    parser=argparse.ArgumentParser(description='Bipartite graph prediction by multi-similarity fusion')
    
    parser.add_argument('--k_dim',type=int,default=8, choices=[8,16,32,64,128,256,512,1024])
    parser.add_argument('--alp',type=float,default=0.5,choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--beta',type=float,default=0.5,choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--num_layer', type=int, default=2, choices=[1,2,3,4,5,6])
    
    
    parser.add_argument('--similarity_att', type=int, default=9, choices=[-1,0,1,2,3,4,5,6,7,8,9],
                        help='-1 for average, 0 for Huang att, 1 for bi att 2 for cross end att, 3 for cosine, 4 for cat average, 5 for A3NCF, 6 for Global Local attention, 7 for # 直接给与linear转换')    
    parser.add_argument('--layer_no', type=int, default=-1, help='single layer (-1) or not (>-1), no greater than num_layer', choices=[-1,0,1,2,3,4,5])
    parser.add_argument('--ind_sim', type=int, default=-1,
                        help='-1 for all similarity, bigger than -1 for travel 0,1,2,...ind_sim-th single similarity')
    
    parser.add_argument('--feature_type', default='dgi', choices=[
                        'one_hot', 'random_uniform', 'random_normal',
                        'dgi',                        
                        'bionev_LINE', 'bionev_SDNE', 'bionev_GAE',
                        'bionev_DeepWalk', 'bionev_node2vec', 'bionev_struc2vec', 
                        'bionev_Laplacian', 'bionev_GF', 'bionev_SVD', 'bionev_HOPE', 'bionev_GraRep'])
    # 数据集名称
    parser.add_argument('--dataName', default='ZhangMDA', 
                        choices=['deepDR_data', 'LuoDTI', 'Enzyme_data', 'ZhangMDA', 'gene_disease_data', 
                                 'GPCR_data', 'IC_data', 'ImpHuman_data', 'LiuDTI', 
                                 'Liu_data', 'LRSSL_data', 'NIMCD1_data', 'NIMCD2_data', 
                                 'Nuclear_data', 'SCMFDD_data', 'SFPEL_data'],
                        help='the data name')
    
    ### case study
    parser.add_argument('--caseStudy', type=int, default=0, help='case study if 1')
    parser.add_argument('--case_x', type=int, default=-1, help='row No. for case study; -1 for all row No.')
    parser.add_argument('--case_y', type=int, default=-1, help='col No. for case study; -1 for all col No.')
    ### tsne_flag cross test
    parser.add_argument('--tsne_flag', type=int, default=0, help='plot tsne figure')
    parser.add_argument('--cross', type=int, default=0, help='cross validate or not')
    parser.add_argument('--test', type=int, default=1, help='test or not')
    
    parser.add_argument('--single_fold', type=int, default=1, help='whether one fold')
    # 交叉验证的折数
    parser.add_argument('--kfold',type=int,default=0,help='index of train or validation k number')
    # 交叉验证的总折数
    parser.add_argument('--nfold',type=int,default=5,choices=[5, 10],help='index of train or validation total number')
    
    parser.add_argument('--prin', type=int, default=1, choices=[0,1])
    parser.add_argument('--result_key', type=str, default='_', help='key word for result')
    
    
    
    parser.add_argument('--hh_fusion', default='add', choices=['cat', 'add'])   ###
    parser.add_argument('--layer_att', type=int, default=0, choices=[0,1])   ###
        
    parser.add_argument('--train_type', default='pretrain', choices=['pretrain','finetuning','one_stage'])   ###
    # if train_type is one_stage, then set the dgi_weight
    parser.add_argument('--dgi_weight', type=float, default=0.0, choices=[0., 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])   ###
    
    # normalize type
    parser.add_argument('--normalize', default='row_sum_one', choices=['col_mean_zero', 'minmax', 'softmax', 'row_sum_one', 'none'])   ###
    parser.add_argument('--init', default='he_normal', choices=['xavier_normal', 'xavier_uniform', 'he_normal', 'he_uniform'])   ###
    parser.add_argument('--func', default='raw', choices=['raw', 'sigmoid', 'tanh', 'relu', 'leaky_relu'])    
    # 模型结构参数    
    parser.add_argument('--score_type', default='cat', choices=['none', 'ncf_linear', 'w','vec', 'cat'])   ###
    parser.add_argument('--sigmoid_flag', type=int, default=1, choices=[0,1])   ###
    parser.add_argument('--lossType',default='MF_all', choices=['MF_all', 'cross_entropy', 'MSE'])   ###
    
    parser.add_argument('--lr', type=float, default=0.0005, choices=[0.001,0.0005], help='the learning rate of GCN')   ###
    parser.add_argument('--weight_decay', type=float, default=0., choices=[0.,1e-8,0.00001,0.0001,0.001,0.01,0.1,1,10])   ###
    parser.add_argument('--dropout', type=float, default=0.1, choices=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])   ###
    ### ### 不常变的参数 ### ###
    
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'], help='the DEVICE of GCN')   ###
    parser.add_argument('--epochs', type=int, default=100, help='number of max epoch')   ###
    parser.add_argument('--patience', type=int, default=10, help='patience')   ###
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')   ###
    parser.add_argument('--seed', type=int, default=1, help='seed')   ###
    parser.add_argument('--seed_model', type=int, default=1, help='seed')   ###
    parser.add_argument('--seed_test', type=int, default=1, help='seed')   ###
    parser.add_argument('--seed_traval', type=int, default=1, help='seed')   ###
    
    ### ### 引用参数 ### ###
    opt = parser.parse_args()
    opt.tra_fold = opt.kfold
    if opt.layer_no > opt.num_layer:
        print('the layer no is greater than num layer')
        sys.exit(1)
    opt.fastmode = False
    return opt