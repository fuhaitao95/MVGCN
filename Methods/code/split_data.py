# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:29:10 2020
@author: FHt
"""


import numpy as np

import os
import scipy.sparse as sp
import sys

from utils.read_raw_data import readData

def splitData(dataY, splitPath, nfold, seed_indent, seed_cross, indentKey='indent', crossKey='cross'):
    # dataY is the association matrix，the split it into k fold, saved separately
    # neg_pos_ratio: the ratio between the negative samples and the positive samples
    # indent_tes_ratio：the ratio of the independent test samples in the dataset
    neg_pos_ratio = 1.0
    indent_tes_ratio = 0.1
    # seed = 1
    
    index_pos = np.array(np.where(dataY == 1))
    index_neg = np.array(np.where(dataY == 0))
    pos_num = len(index_pos[0])
    neg_num = int(pos_num * neg_pos_ratio)
    np.random.seed(seed_indent)
    np.random.shuffle(index_pos.T)
    np.random.seed(seed_indent)
    np.random.shuffle(index_neg.T)
    index_neg = index_neg[:, : neg_num]
    
    indent_tes_pos_num = int(pos_num * indent_tes_ratio)
    indent_tes_neg_num = int(neg_num * indent_tes_ratio)
    indent_tes_index = np.hstack((index_pos[:, : indent_tes_pos_num], index_neg[:, : indent_tes_neg_num]))
    indent_tes_data = np.hstack((indent_tes_index.T, dataY[indent_tes_index[0], indent_tes_index[1]].reshape(-1, 1)))
    np.savetxt(splitPath + indentKey + '_tes_kfold' + str(-1) + '_seed' + str(seed_indent) + '.txt', indent_tes_data, fmt='%d', delimiter=',')
    # '_total represents all edges, only two columns指, the second column added with the node number of the first column
    # ensuring that the numbers are unique
    indent_tes_data_total = indent_tes_data[indent_tes_data[:,-1]==1][:,:-1]
    indent_tes_data_total[:,1]+=dataY.shape[0]
    np.savetxt(splitPath + indentKey + '_tes_kfold' + str(-1) + '_seed' + str(seed_indent) + '_total.txt', indent_tes_data_total, fmt='%d', delimiter=' ')
    
    cross_pos = index_pos[:, indent_tes_pos_num: ]
    cross_neg = index_neg[:, indent_tes_neg_num: ]
    
    temp = np.hstack((cross_pos, cross_neg))
    indent_tra_data = np.hstack((temp.T, dataY[temp[0], temp[1]].reshape(-1, 1)))
    indent_tra_matx = sp.coo_matrix((indent_tra_data[:, 2], (indent_tra_data[:, 0],indent_tra_data[:, 1])), shape=(dataY.shape[0],dataY.shape[1])).toarray()
    indent_tra_data_total = indent_tra_data[indent_tra_data[:, -1]==1][:, :-1]
    indent_tra_data_total[:,1]+=dataY.shape[0]
    # the below codes means there is no explicit validate data, all data only split into train and test data
    np.savetxt(splitPath + indentKey + '_tra_kfold' + str(-1) + '_seed' + str(seed_indent) + '.txt', indent_tra_data, fmt='%d', delimiter=',')
    np.savetxt(splitPath + indentKey + '_tra_kfold' + str(-1) + '_seed' + str(seed_indent) + '_mat.txt', indent_tra_matx, fmt='%d', delimiter=',')
    np.savetxt(splitPath + indentKey + '_tra_kfold' + str(-1) + '_seed' + str(seed_indent) + '_total.txt', indent_tra_data_total, fmt='%d', delimiter=' ')
    # -1 of kfold means the tra_val vs test data, there is no explicit validate data, all data only split into train and test data
    np.savetxt(splitPath + indentKey + '_tra_kfold' + str(-1) + '_seed' + str(seed_indent) + '.txt', indent_tra_data, fmt='%d', delimiter=',')
    np.savetxt(splitPath + indentKey + '_tra_kfold' + str(-1) + '_seed' + str(seed_indent) + '_mat.txt', indent_tra_matx, fmt='%d', delimiter=',')
    np.savetxt(splitPath + indentKey + '_tra_kfold' + str(-1) + '_seed' + str(seed_indent) + '_total.txt', indent_tra_data_total, fmt='%d', delimiter=' ')
    
    np.random.seed(seed_cross)
    np.random.shuffle(cross_pos.T)
    np.random.shuffle(cross_neg.T)
        
    cross_fold_index_pos = np.array([temp % nfold for temp in range(len(cross_pos[0]))])
    cross_fold_index_neg = np.array([temp % nfold for temp in range(len(cross_neg[0]))])
    
    kfold=0
    for kfold in range(nfold):
        cross_tra_fold_pos = cross_pos.T[cross_fold_index_pos != kfold]
        cross_tes_fold_pos = cross_pos.T[cross_fold_index_pos == kfold]
        cross_tra_fold_neg = cross_neg.T[cross_fold_index_neg != kfold]
        cross_tes_fold_neg = cross_neg.T[cross_fold_index_neg == kfold]
        
        cross_tra_fold = np.vstack((cross_tra_fold_pos, cross_tra_fold_neg))
        cross_tes_fold = np.vstack((cross_tes_fold_pos, cross_tes_fold_neg))
        
        cross_tra_data = np.hstack((cross_tra_fold, dataY[cross_tra_fold[:, 0], cross_tra_fold[:, 1]].reshape(-1, 1)))        
        cross_tes_data = np.hstack((cross_tes_fold, dataY[cross_tes_fold[:, 0], cross_tes_fold[:, 1]].reshape(-1, 1)))
        cross_tra_matx = sp.coo_matrix((cross_tra_data[:, 2], (cross_tra_data[:, 0],cross_tra_data[:, 1])), shape=(dataY.shape[0],dataY.shape[1])).toarray()
        cross_test_matx = sp.coo_matrix((cross_tes_data[:, 2], (cross_tes_data[:, 0],cross_tes_data[:, 1])), shape=(dataY.shape[0],dataY.shape[1])).toarray()
        np.savetxt(splitPath + crossKey + '_tra_kfold' + str(kfold) + '_seed' + str(seed_cross) + '.txt', cross_tra_data, fmt='%d', delimiter=',')
        np.savetxt(splitPath + crossKey + '_tes_kfold' + str(kfold) + '_seed' + str(seed_cross) + '.txt', cross_tes_data, fmt='%d', delimiter=',')
        np.savetxt(splitPath + crossKey + '_tra_kfold' + str(kfold) + '_seed' + str(seed_cross) + '_mat.txt', cross_tra_matx, fmt='%d', delimiter=',')
        np.savetxt(splitPath + crossKey + '_tes_kfold' + str(kfold) + '_seed' + str(seed_cross) + '_mat.txt', cross_test_matx, fmt='%d', delimiter=',')
        cross_tra_data_total = cross_tra_data[cross_tra_data[:, -1]==1][:, :-1]
        cross_tes_data_total = cross_tes_data[cross_tes_data[:, -1]==1][:, :-1]
        cross_tra_data_total[:,1]+=dataY.shape[0]
        cross_tes_data_total[:,1]+=dataY.shape[0]
        np.savetxt(splitPath + crossKey + '_tra_kfold' + str(kfold) + '_seed' + str(seed_cross) + '_total.txt', cross_tra_data_total, fmt='%d', delimiter=' ')
        np.savetxt(splitPath + crossKey + '_tes_kfold' + str(kfold) + '_seed' + str(seed_cross) + '_total.txt', cross_tes_data_total, fmt='%d', delimiter=' ')
    
    return

def splitDataMain(nfold, dataName, seed_indent=1, seed_cross=1):
    dataPath = '../../Datasets/'+dataName + '/'
    
    usedDataPath = dataPath+'used_data/'
    splitPath = '_'.join([dataPath+'splitData_TraValTes',str(nfold)+'nfold','seedIndent'+str(seed_indent),'seedCross'+str(seed_cross)+'/'])
    if not os.path.exists(splitPath):
        os.mkdir(splitPath)
    # read association data
    dataY, AAr, bAr, ANet, bNet, names = readData(dataName, usedDataPath)
    # split into k fold
    splitData(dataY, splitPath, nfold, seed_indent, seed_cross)
    return dataY, AAr, bAr, ANet, bNet, names

if __name__ == '__main__':
    '''
    example command：python split_data.py fold_number(int) dataName seed_indent(int) seed_cross(int)
    example: python process_data.py 5 ZhangDDA 1 1
    '''
    argv = sys.argv
    nfold = int(argv[1])
    dataName = argv[2]
    seed_indent = int(argv[3])
    seed_cross = int(argv[4])
    print(argv)
    dataY, AAr, bAr, ANet, bNet, names = splitDataMain(nfold, dataName, seed_indent, seed_cross)
    total_number = dataY.sum()
    sparsity = 1 - total_number / float(dataY.shape[0] * dataY.shape[1])
    print(dataY.shape, total_number, round(sparsity * 100,2))

