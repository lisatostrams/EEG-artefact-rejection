# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:13:34 2018

@author: Lisa Tostrams
"""

import numpy as np
from scipy import signal
import gc
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from pandas import rolling_corr
from scipy.linalg import lstsq


class Validate(object):
    
    def __init__(self, X,Xc,EOG_chans,B=None,peeg=None):
        assert (not (B is None and peeg is None)), "Either true sigal B (for simulated data) or PEEG_Analyse instance (for acquired data) needs to be initialized." 
        mask = np.ones(len(X),dtype=bool)
        mask[EOG_chans] = False
        if(not (peeg is None)): #exclude trigger channel
            mask[-1] = False

        self.O = X[EOG_chans]
        self.X = X[mask,:]
        self.Xc = Xc[mask,:]
        self.B = B
#        
#        corrH,corrV = self.corr(X,21,22)
#        OA = self.annotate(X,21,22)
#        self.show_OA(peeg,X,OA)
##        
#        print(*['corr EOG_H,{} = {}'.format(peeg.signalLabels[i],np.mean(corrH,1)[i]) for i in range(21)],sep='\n')
#        print()
#        print(*['corr EOG_V,{} = {}'.format(peeg.signalLabels[i],np.mean(corrV,1)[i]) for i in range(21)],sep='\n')
#            
#    
    def regression(self):
        X = np.copy(self.X).T
        Xc = np.copy(self.Xc).T
        O = np.copy(self.O).T
        beta_X = np.zeros([len(X.T),2])
        for i in range(len(X.T)):    
            p,res,rnk,s = lstsq(O,X[:,i])
            beta_X[i,:] = p
        beta_Xc = np.zeros([len(Xc.T),2])
        for i in range(len(Xc.T)):    
            p,res,rnk,s = lstsq(O,Xc[:,i])
            beta_Xc[i,:] = p
        return beta_X,beta_Xc
    
    def NMSE(self):
        assert not (self.B is None), 'Initialize B (ground truth) first to calculate NMSE'
        return sum(((self.Xc-self.B)**2).T)/sum((self.B**2).T)
    
    
    def corr(X,idx_h,idx_v):
        n_signals = len(X)
        ts = len(X.T)
        corrH = np.zeros([n_signals-3,ts])
        corrV = np.zeros([n_signals-3,ts])
        EOG_H = pd.Series(X[idx_h])
        EOG_V = pd.Series(X[idx_v])
        for sign in range(n_signals-3):
            sig = pd.Series(X[sign])
            corrH[sign] = rolling_corr(EOG_H,sig,6*1024,min_periods=100,center=True).values
            corrV[sign] = rolling_corr(EOG_V,sig,6*1024,min_periods=100,center=True).values 
        del EOG_H
        del EOG_V
        del sig
        return corrH,corrV


    def annotate(X,idx_h,idx_v,ccoef=0.5):
        corr_H,corr_V = corr(X,idx_h,idx_v)
        OA = ((abs(corr_H) > ccoef) | (abs(corr_V) > ccoef))
        return OA

    def show_OA(peeg, X,OA):
        plt.close('all')
        f, axarr = plt.subplots(peeg.nSignals, 1, figsize=(12,30),sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.05, hspace = 0.3)
        t = np.arange(0, peeg.nSamples[0]/peeg.fs[0], peeg.Ts[0]) 
        axarr[0].set_title('$X^s(t)$')
        for i in range(peeg.nSignals):
            axarr[i].set_ylabel(peeg.signalLabels[i])
            axarr[i].plot(t, X[i])
            if(i < 21):
                trans = mtransforms.blended_transform_factory(axarr[i].transData, axarr[i].transAxes)
                axarr[i].fill_between(t, 0, 1, where=OA[i,:], facecolor='red', alpha=0.5, transform=trans)
        f.tight_layout()
        plt.show()