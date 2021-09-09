# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 20:12:03 2020

@author: xinxi
"""

import numpy as np

def normalizeSoft(feature):
    f_exp = np.exp(feature)
    f_sum = np.tile(f_exp.sum(1).reshape(-1,1),(1, feature.shape[1]))
    feat_norm = f_exp / f_sum   
    
    return feat_norm
    
def normalizeRow(mx):
    """Row-normalize matrix"""
    rowsum = np.array(mx).sum(1)
    rowsum[rowsum==0]=1.
    r_inv = np.power(rowsum, -1).flatten()
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def normalizeRowCol(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj).sum(1)
    rowsum[rowsum==0]=1.
    d_inv_sqrt_row = np.power(rowsum, -0.5).flatten()
    d_mat_inv_sqrt_row = np.diag(d_inv_sqrt_row)
    
    colsum=np.array(adj).sum(0)
    colsum[colsum==0]=1
    d_inv_sqrt_col=np.power(colsum,-0.5).flatten()
    d_mat_inv_sqrt_col=np.diag(d_inv_sqrt_col)
    
    result=d_mat_inv_sqrt_row.dot(adj).dot(d_mat_inv_sqrt_col)
    return result
def normSim(sim_matrix):
    for i in range(len(sim_matrix)):
        for j in range(len(sim_matrix[i])):
            sim_matrix[i,j,j] = 0.0
            pass
        sim_matrix[i] = (sim_matrix[i] + sim_matrix[i].T) / 2.0
        sim_matrix[i] = normalizeRowCol(sim_matrix[i])
    return sim_matrix