#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:59:49 2020

@author: fht
"""

import numpy as np
import scipy.sparse as sp
from .similarity import get_Jaccard_Similarity



def diagZeroAdj(ar):
    for i in range(len(ar)):
        for j in range(ar[i].shape[0]):
            ar[i][j,j] = 0
    return ar

def get_profile_sim(sim,data,row,col):
    mat = sp.coo_matrix((data[:,2],(data[:,0],data[:,1])),
                        shape=(row,col),dtype=np.int).toarray()
    sim_profile = np.array(get_Jaccard_Similarity(mat))
    sim_ls=[]
    for item in sim:
        sim_ls.append(item)
    sim_ls.append(sim_profile)
    sim = np.array(sim_ls)    
    return sim