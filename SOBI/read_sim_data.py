# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:40:30 2017

@author: Lima
"""
import numpy as np
from scipy.io import loadmat

def import_data():
    B = loadmat('Data/Pure_Data.mat')
    X = loadmat('Data/Contaminated_Data.mat')
    HEOG = loadmat('Data/HEOG.mat')
    VEOG = loadmat('Data/VEOG.mat')
    electrodes = "FP1,FP2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T3,T4,T5,T6,Fz,Cz,Pz".split(sep=',')

    n = len(B)-3
    Btmp = {}
    Xtmp = {}
    HEOGtmp = {}
    VEOGtmp = {}
    for i in range(1,n+1):
        Xtmp['id{}'.format(i)] = X['sim{}_con'.format(i)]
        HEOGtmp['id{}'.format(i)] = HEOG['heog_{}'.format(i)]
        VEOGtmp['id{}'.format(i)] = VEOG['veog_{}'.format(i)]
        Btmp['id{}'.format(i)] = B['sim{}_resampled'.format(i)]
     
    return Btmp, Xtmp, HEOGtmp, VEOGtmp, electrodes
    