#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 02:35:54 2021

@author: FHt
"""

import numpy as np
import os
import sys
import torch

from utils.process_para import optPara
from model_travaltes import trainTestMain
from utils.write_function import write_result,write_score
from utils.get_data import getTraValTesData


def cross5CV(opt):
    opt.cross_indent = 'cross'
    nfold = opt.nfold
    dataName = opt.dataName
    crossKey = opt.crossKey
    seed_cross = opt.seed_cross
    seed_indent = opt.seed_indent
    exp_name = opt.exp_name
    
    dataPath = '../../Datasets/'+dataName+'/'
    opt.usedDataPath = usedDataPath = '../../Datasets/'+dataName+'/'+'used_data/'
    opt.splitPath = splitPath = '_'.join([dataPath+'splitData_TraValTes', str(nfold)+'nfold', 'seedIndent'+str(seed_indent), 'seedCross'+str(seed_cross)+'/'])
    

    lossType = opt.lossType
    result_key = opt.result_key
    # 结果文件参数
    opt.resultTxt = resultTxt = dataName + '/' + '_'.join([exp_name,dataName,'resultTxt',lossType,crossKey,result_key]) + '.csv'
    
    result = np.zeros((nfold+2,8))
    for kfold in range(nfold):
        opt.kfold = kfold
        tra_name = splitPath + crossKey + '_tra_kfold' + str(kfold) + '_seed' + str(opt.seed_indent) + '.txt'
        tes_name = splitPath + crossKey + '_tes_kfold' + str(kfold) + '_seed' + str(opt.seed_indent) + '.txt'
        val_name = tes_name
        sim_A, sim_b, tra_array, val_array, tes_array = getTraValTesData(dataName, usedDataPath, tra_name, val_name, tes_name)
        test_label, test_score, criteria_result, model_max, F_u, F_i = trainTestMain(opt, sim_A, sim_b, tra_array, val_array, tes_array)
        
        result[kfold,0]=kfold; result[kfold,1:]=np.array(criteria_result)
        
        fileName = resultTxt.rstrip('v').rstrip('s').rstrip('.c') + '_label_score_kfold'+str(kfold)+'.csv'
        write_score(test_label, test_score, fileName, opt)
        
    result[-2] = np.mean(result[:-2],axis=0)
    result[-1] = np.std(result[:-2],axis=0)
    
    write_result(result, resultTxt, opt)
    
    return test_label, test_score, criteria_result, model_max, F_u, F_i

def indentTraTes(opt):
    opt.cross_indent = 'indent'
    nfold = opt.nfold
    dataName = opt.dataName
    indentKey = opt.indentKey
    seed_cross = opt.seed_cross
    seed_indent = opt.seed_indent
    exp_name = opt.exp_name
    
    dataPath = '../../Datasets/'+dataName+'/'
    opt.usedDataPath = usedDataPath = dataPath+'used_data/'
    opt.splitPath = splitPath = '_'.join([dataPath+'splitData_TraValTes',str(nfold)+'nfold','seedIndent'+str(seed_indent),'seedCross'+str(seed_cross)+'/'])
    

    lossType = opt.lossType
    result_key = opt.result_key
    # the output file path and name
    opt.resultTxt = resultTxt = dataName + '/' + '_'.join([exp_name,dataName,'resultTxt',lossType,indentKey,result_key]) + '.csv'
    
    opt.kfold = kfold = -1
    tra_name = splitPath + indentKey + '_tra_kfold' + str(kfold) + '_seed' + str(opt.seed_indent) + '.txt'
    tes_name = splitPath + indentKey + '_tes_kfold' + str(kfold) + '_seed' + str(opt.seed_indent) + '.txt'
    val_name = tes_name
    
    sim_A, sim_b, tra_array, val_array, tes_array = getTraValTesData(dataName, usedDataPath, tra_name, val_name, tes_name)
    
    test_label, test_score, criteria_result, model_max, F_u, F_i = trainTestMain(opt, sim_A, sim_b, tra_array, val_array, tes_array)
    
    fileName = resultTxt.rstrip('v').rstrip('s').rstrip('.c') + '_label_score_kfold'+str(kfold)+'.csv'
    write_score(test_label, test_score, fileName, opt)
    
    result = np.zeros((1,8)); result[0,1:] = np.array(criteria_result)
    write_result(result, resultTxt, opt)
    
    return test_label, test_score, criteria_result, model_max, F_u, F_i




if __name__ == '__main__':
    opt = optPara()    
    cross5CV(opt)
    test_label, test_score, criteria_result, model_max, F_u, F_i = indentTraTes(opt)
    