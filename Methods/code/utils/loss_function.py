# -*- coding: utf-8 -*-
"""


@author: Fu Haitao
"""


import torch
import torch.nn.functional as F


def lossF(lossType, predictions, targets):
    if lossType == 'cross_entropy':
        pos_weight = 1.
        neg_weight = 1.
        weightTensor = torch.zeros(len(targets))
        weightTensor[targets == 1] = pos_weight
        weightTensor[targets == 0] = neg_weight
        if (predictions.min() < 0) | (predictions.max() > 1):
            losses = F.binary_cross_entropy_with_logits(predictions.double(), targets.double(), weight=weightTensor)
        else:
            losses = F.binary_cross_entropy(predictions.double(), targets.double(), weight=weightTensor)
    elif lossType == 'MF_all':
        losses = torch.pow((predictions - targets), 2).mean()
    elif lossType == 'MSE':
        losses = F.mse_loss(predictions, targets)
    return losses
