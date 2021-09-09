#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 03:12:49 2021

@author: fht
"""



import numpy as np



from utils.read_raw_data import readData

def getTraValTesData(dataName, usedDataPath, tra_name, val_name, tes_name):
    dataY, sim_A, sim_b, ANet, bNet, names = readData(dataName, usedDataPath)
    tra_array = np.loadtxt(tra_name, delimiter=',')
    val_array = np.loadtxt(val_name, delimiter=',')
    tes_array = np.loadtxt(tes_name, delimiter=',')
    return sim_A, sim_b, tra_array, val_array, tes_array


        
        