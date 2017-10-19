# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:23:41 2017

@author: Lima
"""

import sim_data
import SOBI

Data = sim_data.SimData()


def NMSE(C):
    B = Data.B['id{}'.format(id)]
    Ct = C[0:19,:]
    return sum(((Ct-B)**2).T)/sum((B**2).T)
    

#%%
for id in range(55):
    