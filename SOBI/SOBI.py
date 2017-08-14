# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:39:17 2017

@author: Lima
"""

import scipy
from scipy import signal
import numpy as np
from sim_data import SimData



Data = SimData()

import matplotlib.pyplot as plt

X = Data.X['id1']
R = signal.fftconvolve(X,X, mode='full')

def cross_corr(tau):

    s=0

def SOBI(id, tau):
    components = []



    return components