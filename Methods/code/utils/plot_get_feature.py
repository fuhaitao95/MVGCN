# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 04:00:01 2021

@author: FHt
"""

import numpy as np

def plt_tsne(opt, fig_marker, X, label):
    import matplotlib.pyplot as plt
    from sklearn import manifold
    import pandas as pd
    np.random.seed(1)
    pos_label = label==1
    neg_label = label==0
    plt.rcParams['font.sans-serif'] = ['Arial']
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=1)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    np.savetxt(opt.dataName + '/' + fig_marker+'_2d.csv', np.hstack((X_norm,label.reshape(-1,1))), fmt='%.4f', delimiter=',')
    
    plt.figure(figsize=(10, 10))
    plt.tick_params(labelsize=30)
    plt.scatter(X_norm[:,0][pos_label],X_norm[:,1][pos_label],edgecolors=('black'),color='r', label='pos')
    plt.scatter(X_norm[:,0][neg_label],X_norm[:,1][neg_label],edgecolors=('black'),color='g', label='neg')
    legend_font = {'family': 'Arial', 'style': 'normal','size': 20,  'weight': "bold"}
    plt.legend(loc='lower right', prop=legend_font)
    plt.savefig(opt.dataName + '/' + fig_marker+'.tif',dpi=350)
    return