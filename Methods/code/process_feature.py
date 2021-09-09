# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:17:32 2020
modified on Mon Jan 4 20:05 2021
modified on Mon Feb 03 17:16 2021
@author: xinxi
"""

import numpy as np
import os
import sys
import torch



seed = 1
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def getFeature(opt, traiY, row_sim_matrix, col_sim_matrix):

    normalizeType = opt.normalizeType
    splitPath = opt.splitPath
    nfold = opt.nfold
    kfold = opt.kfold
    
    cudaFlag = opt.cudaFlag
    cross_indent = opt.cross_indent

    
    seed = 1
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print('getting initial node embeddings, it may take dozens of miniutes, please wait patiently......')
    
    
    from utils.process_dgi import dgi_embed
    opt.in_features = in_features = traiY.shape[0] + traiY.shape[1]
    
    embedding_file = splitPath+'DGI_embedding/'
    if not os.path.exists(embedding_file):
        os.makedirs(embedding_file)
    embedding_suffix = '_dim'+str(in_features)+'_'+str(nfold)+'nfold'+'_kfold'+str(kfold)+'.txt'
    output_file = embedding_file+cross_indent+'_dgi'+embedding_suffix
    if not os.path.exists(output_file):            
        association = np.vstack((np.hstack((np.zeros((traiY.shape[0],traiY.shape[0]),dtype=np.float32),
                                            traiY)),
                                 np.hstack((traiY.T,
                                            np.zeros((traiY.shape[1],traiY.shape[1]),dtype=np.float32)
                                            ))
                                 ))
        if not os.path.exists(splitPath+'dgi_model/'):
            os.mkdir(splitPath+'dgi_model/')
        model_name = splitPath + '/dgi_model/'+cross_indent+'_bestDGI_in_features'+str(in_features)+'_'+str(nfold)+'nfold_'+'kfold'+str(kfold)+'.pkl'
        feat_array = dgi_embed(association, np.eye(in_features), in_features,
                               model_name, 
                               cudaFlag)
        np.savetxt(output_file, feat_array)
    else:
        feat_array = np.loadtxt(output_file)            
    
    
    ### normalize ###
    if normalizeType == 'col_mean_zero':
        from sklearn import preprocessing
        feat_array = preprocessing.scale(feat_array)
    elif normalizeType == 'minmax':
        from sklearn.preprocessing import minmax_scale
        feat_array = minmax_scale(feat_array)
    elif normalizeType == 'softmax':
        from utils.normalization import normalizeSoft
        feat_array = normalizeSoft(feat_array)
    elif normalizeType == 'row_sum_one':
        from utils.normalization import normalizeRow
        feat_array = normalizeRow(feat_array)
    elif normalizeType == 'none':
        pass
    else:
        print('parameter normalizeType is wrong')
        print(normalizeType)
        sys.exit()
    F_u_temp = feat_array[: traiY.shape[0]]
    F_i_temp = feat_array[traiY.shape[0]: ]
    return np.array(F_u_temp), np.array(F_i_temp)