#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 03:56:39 2021

@author: fht
"""

import torch

from torch.utils import data

class dataClass(data.Dataset):
    def __init__(self, data_list_k):        
        self.idx0 = torch.LongTensor(data_list_k[:,0])
        self.idx1 = torch.LongTensor(data_list_k[:,1])
        self.labels = torch.FloatTensor(data_list_k[:,2])
    def __len__(self):
        return len(self.idx0)
    def __getitem__(self, index):
        index0 = self.idx0[index]
        index1 = self.idx1[index]
        y = self.labels[index]
        return index0, index1, y    
def getLoader(batch_size, data_list_k):
    params = {'batch_size': batch_size,
              'shuffle': True,
              'drop_last' : False}
    data_set = dataClass(data_list_k)
    loader = data.DataLoader(data_set, **params)
    return loader