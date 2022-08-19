# -*- coding: utf-8 -*-
"""

@author: Fu Haitao
"""

import numpy as np

import sys
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_
import torch as t

from utils.NIPLayer import NIP

seed = 1
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class MF(torch.nn.Module):
    def __init__(self, opt, num_sim):
        super(MF, self).__init__()
        
        seed = opt.seed
        initFunType = opt.initFunType
        in_features = opt.in_features
        mid_dim = opt.mid_dim        
        num_layer = opt.num_layer
        actType = opt.actType
        
        self.row_num = opt.row_num
        self.dropProb = dropProb = opt.dropProb
        
        if opt.cudaFlag:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.att_embed = Parameter(t.FloatTensor(np.random.rand(opt.num_layer)))
        self.get_initF(initFunType)
        self.encoder_init(num_sim, num_layer, in_features, mid_dim, actType, dropProb)
        self.att_init(mid_dim, num_sim)
        self.decoder_init(mid_dim)
        return
    
    def get_initF(self, initFunType='kaiming_normal'):
        if initFunType == 'xavier_normal':
            initF = xavier_normal_
        elif initFunType == 'xavier_uniform':
            initF = xavier_uniform_
        elif initFunType == 'kaiming_normal':
            initF = kaiming_normal_
        elif initFunType == 'kaiming_uniform':
            initF = kaiming_uniform_
        self.initF = initF
        return
    
    def encoder_init(self, num_sim, num_layer, in_features, mid_dim, actType, dropProb):
        # the 0th layer feature transformation parameters
        self.U = Parameter(torch.FloatTensor(num_sim, in_features, mid_dim))
        self.initF(self.U.data)        
        self.V = Parameter(torch.FloatTensor(num_sim, in_features, mid_dim))
        self.initF(self.V.data)        
        # dropout for E_0
        self.hidden_dropout0 = torch.nn.Dropout(dropProb)
        # NIPlayer message passing
        self.NIPLayer_ls = nn.ModuleList()
        for i_layer in range(num_layer):
            self.NIPLayer_ls.append(nn.ModuleList())
            for j_num_sim in range(num_sim):
                temp = NIP(mid_dim, mid_dim, self.initF, actType, dropProb)
                self.NIPLayer_ls[i_layer].append(temp)
        return
    
    def att_init(self, mid_dim, num_sim):
        # A3NCF attention
        self.att_cat_X1 = nn.Linear(mid_dim * num_sim, 1)
        self.att_cat_X2 = nn.Linear(1, mid_dim)
        self.att_cat_Y1 = nn.Linear(mid_dim * num_sim, 1)
        self.att_cat_Y2 = nn.Linear(1, mid_dim)
        att_ls = [self.att_cat_X1, self.att_cat_X2, self.att_cat_Y1, self.att_cat_Y2]
        for att in att_ls:
            nn.init.constant_(att.weight.data, 1)
            nn.init.constant_(att.bias.data, 0)
        # 直接权重softmax对应求和
        # self.x_att = Parameter(torch.FloatTensor(np.zeros(num_sim,)))
        # self.y_att = Parameter(torch.FloatTensor(np.zeros(num_sim,)))
        self.x_att = Parameter(torch.FloatTensor(np.ones(num_sim,)))
        self.y_att = Parameter(torch.FloatTensor(np.ones(num_sim,)))        
        return
    
    def decoder_init(self, mid_dim):
        # decoder MLP
        self.decoder0 = nn.Linear(mid_dim * 2, mid_dim)
        self.decoder1 = nn.Linear(mid_dim, int(mid_dim/2))
        self.decoder2 = nn.Linear(int(mid_dim/2), 1)
        # decoder parameters: bilinear
        self.XYW = Parameter(torch.FloatTensor(mid_dim, mid_dim))
        self.initF(self.XYW.data)
        # decoder parameters: vector
        self.W_vec = Parameter(torch.FloatTensor(1, mid_dim))
        self.initF(self.W_vec.data)
        return
    def decoder(self, X_embed, Y_embed, decoder_type, sigmoid_flag):
        if decoder_type == 'none':
            pred = torch.mul(X_embed, Y_embed).sum(1)
        elif decoder_type == 'ncf_linear':
            o = torch.mul(X_embed, Y_embed)
            o = F.leaky_relu(self.decoder1(o))
            pred = self.decoder2(o).flatten()
        elif decoder_type == 'w':
            pred = torch.mul(torch.matmul(X_embed, self.XYW), Y_embed).sum(1)
        elif decoder_type == 'vec':
            pred = torch.mul(X_embed, torch.mul(Y_embed, self.W_vec)).sum(1)
        elif decoder_type == 'cat':
            feat = torch.cat((X_embed, Y_embed), dim=1)
            o = F.leaky_relu(self.decoder0(feat))
            o = F.leaky_relu(self.decoder1(o))
            self.tsneX = o.cpu().detach().numpy()
            pred = self.decoder2(o).flatten()
        if sigmoid_flag:
            pred = nn.Sigmoid()(pred)
        return pred
        
    def getEmbedMid(self, E_ls, num_layer):

        
        embedMid = torch.cat([item.unsqueeze(0) for item in E_ls],0).mean(0)
        
        return embedMid
    
    def A3NCFAtt(self, X, Y, idx0, idx1):
            # cat linear transformation: A3NCF
            X_sample = X[:,idx0,:].transpose(0,1) # X_sample: sample, feat_num, feat_dim
            Y_sample = Y[:,idx1,:].transpose(0,1)
            
            X_cat = torch.cat([item for item in X_sample.transpose(0,1)],dim=1)
            Y_cat = torch.cat([item for item in Y_sample.transpose(0,1)],dim=1)
            
            att_X = F.softmax(self.att_cat_X2(F.leaky_relu(self.att_cat_X1(X_cat))), dim=-1)
            att_Y = F.softmax(self.att_cat_Y2(F.leaky_relu(self.att_cat_Y1(Y_cat))), dim=-1)
            
            # X_embed = torch.mul(X_sample.sum(1), att_X)
            # Y_embed = torch.mul(Y_sample.sum(1), att_Y)
            X_embed = torch.mul(X_sample.mean(1), att_X) * X_sample.shape[1]
            Y_embed = torch.mul(Y_sample.mean(1), att_Y) * X_sample.shape[1]
            return X_embed, Y_embed
    def directAtt(self, X, Y, idx0, idx1):
        # 直接给予权重 direct attention mechanism
        X_sample = X[:,idx0,:].transpose(0,1) # X_sample: sample, feat_num, feat_dim
        Y_sample = Y[:,idx1,:].transpose(0,1)
        
        X_att = F.softmax(self.x_att, dim=0)
        X_embed = torch.cat([(X_att[tt] * X_sample[:, tt, :]).unsqueeze(1) for tt in range(X_sample.shape[1])], dim=1).sum(1)
        
        Y_att = F.softmax(self.y_att, dim=0)
        Y_embed = torch.cat([(Y_att[tt] * Y_sample[:, tt, :]).unsqueeze(1) for tt in range(Y_sample.shape[1])], dim=1).sum(1)
        return X_embed, Y_embed
    def getSimAtt(self, X, Y, idx0, idx1):
        X_embed_A3, Y_embed_A3 = self.A3NCFAtt(X, Y, idx0, idx1)
        X_embed_di, Y_embed_di = self.directAtt(X, Y, idx0, idx1)
        X_embed = 0.5 * (X_embed_A3 + X_embed_di)
        Y_embed = 0.5 * (Y_embed_A3 + Y_embed_di)
        
        return X_embed, Y_embed

    def forward(self, opt, F_u, F_i, idx0, idx1, adj_net_homo, adj_net_hete):
        num_layer = opt.num_layer
        
        decoder_type = opt.decoder_type
        sigmoid_flag = opt.sigmoid_flag
        alp = opt.alp
        beta = opt.beta
        NIPFusionType = opt.NIPFusionType
        num_sim = opt.num_sim
        
        ## F_u.shape: (user_sample_total_num, feat_dim)
        ## F_i.shape: (item_sample_total_num, feat_dim)
        ## idx0, idx1: batch_size, number of train or test
        ## adj_net_homo, adj_net_hete: shape: (user_sim_num * item_sim_num, user_total_num + item_total_num, user_total_num + item_total_num)
        
        feat_u, feat_i = F_u, F_i
        
        X_u = t.matmul(feat_u, self.U)
        X_i = t.matmul(feat_i, self.V)
        E_0 = t.cat((X_u, X_i), 1)
        E_0 = F.dropout(E_0, p=self.dropProb, training=self.training)
        
        E_ls = [E_0]
        for i_layer in range(num_layer):
            E_ls.append(torch.cat([self.NIPLayer_ls[i_layer][j_sim](E_ls[i_layer][j_sim], adj_net_homo[j_sim], adj_net_hete[j_sim], self.row_num, NIPFusionType, alp, beta).unsqueeze(0) for j_sim in range(num_sim)],0))
                
        ### Weighted summation or average
        embedMid = self.getEmbedMid(E_ls, num_layer)
        
        X = embedMid[:, :self.row_num, :]
        Y = embedMid[:, self.row_num:, :]
        self.X_case = X.cpu().detach()
        self.Y_case = Y.cpu().detach()
        X_embed, Y_embed = self.getSimAtt(X, Y, idx0, idx1)
        
        pred = self.decoder(X_embed, Y_embed, decoder_type, sigmoid_flag)
        return pred
    
