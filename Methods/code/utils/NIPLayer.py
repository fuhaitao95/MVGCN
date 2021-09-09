# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 07:47:41 2020

@author: xinxi
"""

import sys
import torch
import torch as t
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
seed = 1
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.manual_seed(seed)
class NIP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, initF, actType, dropout=0.1):
        super(NIP, self).__init__()
        seed = 1
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.init_weight(initF, in_dim, out_dim)
        
        self.linear = torch.nn.Linear(2*out_dim, out_dim)
        self.dropout = dropout
        self.hidden_dropout = torch.nn.Dropout(dropout)        
        self.act = self.get_act(actType)
        
    def init_weight(self, initF, in_dim, out_dim):
        self.W_u_k = Parameter(torch.FloatTensor(in_dim, out_dim))
        initF(self.W_u_k.data)
        
        self.W_i_k = Parameter(torch.FloatTensor(in_dim, out_dim))
        initF(self.W_i_k.data)
        return

    def get_act(self, actType):
        if actType == 'raw':
            act = lambda x: x
        elif actType == 'sigmoid':
            act = torch.sigmoid
        elif actType == 'tanh':
            act = torch.tanh
        elif actType == 'relu':
            act = F.relu
        elif actType == 'leaky_relu':
            act = F.leaky_relu
        else:
            print('The activation function is wrong! ')
            sys.exit(1)
        return act
    def forward(self, E_k_0, adj_net_ho, adj_net_he, A_num, hh_fusion, alp, bet):
        E_k_trans = t.cat((t.matmul(E_k_0[: A_num, :], self.W_u_k),
                           t.matmul(E_k_0[A_num: , :], self.W_i_k)), 0)
        if hh_fusion=='cat':
            E_k_1 = self.linear(t.cat((t.matmul(adj_net_ho, E_k_0),
                                          t.matmul(adj_net_he, E_k_trans)),
                                         1)
                                   )
        elif hh_fusion == 'add':
            homo_info = t.matmul(adj_net_ho, E_k_0)
            hete_info = t.matmul(adj_net_he, E_k_trans)
            E_A_info = (alp*homo_info[:A_num,:]+(1-alp)*hete_info[:A_num,:])
            E_b_info = (bet*homo_info[A_num:,:]+(1-bet)*hete_info[A_num:,:])
            E_k_1 = torch.cat((E_A_info, E_b_info),0)
                        
        else:
            print('the homo het fusion type is wrong')
            sys.exit(1)
        E_k_1 = self.act(E_k_1)
        E_k_1 = F.dropout(E_k_1, p=self.dropout, training=self.training)
        return E_k_1