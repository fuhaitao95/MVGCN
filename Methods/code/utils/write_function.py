#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:53:22 2021

@author: fht
"""


import numpy as np
def write_score(test_label, test_score, fileName, opt):
    test_label, test_score = np.array(test_label), np.array(test_score)
    result = np.vstack((test_label, test_score)).T
    with open(fileName, 'a') as fobj:
        [fobj.write(key+','+str(value)+',') for key, value in vars(opt).items()]
        fobj.write('\n')
    with open(fileName, 'ab') as fobj:
        np.savetxt(fobj, result, fmt='%.4f', delimiter=',')
        fobj.write(b'\n\n')
    return

def write_result(result, resultTxt, opt):
    with open(resultTxt, 'a') as fobj:
        [fobj.write(key+','+str(value)+',') for key, value in vars(opt).items()]
        fobj.write('\n')
        [([fobj.write(str(round(temp,4))+',') for temp in criteria_result],fobj.write('\n')) for criteria_result in result]        
        fobj.write('\n\n')
    return