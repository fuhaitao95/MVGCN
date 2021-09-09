#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:29:36 2021

@author: fht
"""


from model_MF_main import MF_HF_main



def testing(opt, sim_A, sim_b, tra_list, tes_list):
    test_label, test_score, criteria_result, model_max, F_u, F_i = MF_HF_main(opt, sim_A, sim_b, tra_list, tes_list, tes_list)
    with open(opt.resultTxt, 'a') as fobj:
        fobj.write('\n\n')
        fobj.write('test   ,')
        [fobj.write(str(round(temp,4))+',') for temp in criteria_result]
        [fobj.write(key+','+str(value)+',') for key, value in vars(opt).items()]
        fobj.write('\n')
    return test_label, test_score, criteria_result, model_max, F_u, F_i