# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:40:30 2017

@author: Lima
"""
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


class SimData:
    B = {}
    X = {}
    VEOG = {}
    HEOG = {}
    electrodes = "FP1,FP2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T3,T4,T5,T6,Fz,Cz,Pz".split(sep=',')
    t=0
    n=0
    
    def __init__(self):
        B = loadmat('Data/Pure_Data.mat')
        X = loadmat('Data/Contaminated_Data.mat')
        HEOG = loadmat('Data/HEOG.mat')
        VEOG = loadmat('Data/VEOG.mat')
        
    
        self.n = len(B)-3
        Btmp = {}
        Xtmp = {}
        HEOGtmp = {}
        VEOGtmp = {}
        for i in range(1,self.n+1):
            Xtmp['id{}'.format(i)] = X['sim{}_con'.format(i)]
            HEOGtmp['id{}'.format(i)] = HEOG['heog_{}'.format(i)]
            VEOGtmp['id{}'.format(i)] = VEOG['veog_{}'.format(i)]
            Btmp['id{}'.format(i)] = B['sim{}_resampled'.format(i)]
        
        self.B = Btmp
        self.X = Xtmp
        self.HEOG = HEOGtmp
        self.VEOG = VEOGtmp
        
        
    def plot_subject(self, id, save=False):
        plt.close('all')
        f, axarr = plt.subplots(21, 2, figsize=(12,30),sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.05, hspace = 0.3)
        self.t = np.arange(0, (len(self.X['id{}'.format(id)][0]-1))/200, 1/200) 
        axarr[0,0].set_title('$B^s(t)$')
        axarr[0,1].set_title('$X^s(t)$')
        for i in range(19):
            axarr[i,0].set_ylabel(self.electrodes[i])
            axarr[i,0].plot(self.t, self.B['id{}'.format(id)][i])
            axarr[i,1].plot(self.t, self.X['id{}'.format(id)][i])
        axarr[19,0].set_title('$O^s(t)$')
        axarr[19,0].set_ylabel('HEOG')
        axarr[19,0].plot(self.t, self.HEOG['id{}'.format(id)][0])
        axarr[19,1].axis('off')
        axarr[20,0].set_ylabel('VEOG')
        axarr[20,0].plot(self.t, self.VEOG['id{}'.format(id)][0])
        axarr[20,1].axis('off')
        f.tight_layout()
       
        if(save):
            f.savefig('Figures/BandX/simulatedB_X_id{}.png'.format(id),dpi=300)
            plt.close('all')
        
            
     
        
        
#    
