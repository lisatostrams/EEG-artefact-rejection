# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:39:17 2017

@author: Lima
"""

import scipy
from scipy import signal
import numpy as np
from sim_data import SimData
from joint_diagonalizer import jacobi_angles
import time

Data = SimData()

import matplotlib.pyplot as plt


#%%
X = np.concatenate((Data.X['id1'],Data.HEOG['id1'], Data.VEOG['id1']))
nrlags = 100
ts = len(X.T)
taus = np.unique([int(x) for x in scipy.stats.truncnorm.rvs((0 - ts) / (ts/3), ((ts*2)-1 - ts) / (ts/3), loc=ts, scale=ts/3, size=nrlags)])


#plot_sources(X,X)
#%%
def svd_whiten(X):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    X_white = np.dot(U, Vt)
    return X_white, U, s, Vt



def cross_corr(X):
    n_signals = len(X)
    ts = len(X.T)
    OUT = np.zeros([(2*ts)-1,n_signals,n_signals])
    for first in range(n_signals):
        for second in range(n_signals):
            OUT[:,second,first]= signal.fftconvolve(X[first], X[second], mode='full')  

    return OUT
    
    
X, U, s, Vt = svd_whiten(X)
R_tau = cross_corr(X)

R_tau_ = R_tau[taus] 

#%%
start_time = time.time()
W, _,_ = jacobi_angles(R_tau_)
S = np.dot(W.T,X)
print("--- {:.2f} seconds ---".format(time.time() - start_time))
#%%

def plot_sources(S, X, save=False, flipped=False, corrected=False):
    f, axarr = plt.subplots(21, 2, figsize=(12,30),sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.05, hspace = 0.3)
    t = np.arange(0, (len(X[0]-1))/200, 1/200) 
    axarr[0,0].set_title('$X^s(t)$')
    axarr[0,1].set_title('$S^s(t)$')
    if(corrected):
        axarr[0,0].set_title('$C^s(t)$')
        axarr[0,1].set_title('$S^s(t)$ corrected')
        
    for i in range(19):
        axarr[i,0].set_ylabel(Data.electrodes[i])
        axarr[i,0].plot(t, X[i])
        axarr[i,1].plot(t, S[i])
    
   
    axarr[19,0].set_ylabel('HEOG')
    axarr[19,0].plot(t, X[19])
    axarr[19,1].plot(t, S[19])
    axarr[20,0].set_ylabel('VEOG')
    axarr[20,0].plot(t, X[20])
    axarr[20,1].plot(t, S[20])
    f.tight_layout()
   
    if(save and not flipped and not corrected):
        f.savefig('Figures/Sources/X_S_{}.png'.format(1),dpi=300)
        #plt.close('all')
    if(save and flipped):
        f.savefig('Figures/Sources/X_S_{}_flippedeog.png'.format(1),dpi=300)
    if(save and corrected):
        f.savefig('Figures/Sources/C_S_{}_corrected.png'.format(1),dpi=300)
        
        
    f.show()

#%%
plot_sources(S, X)

#%%
Xf = np.concatenate((Data.X['id1'],-Data.HEOG['id1'], -Data.VEOG['id1']))
Xf,_,_,_ = svd_whiten(Xf)
R_tauf = cross_corr(Xf)
R_tau_f = R_tauf[taus] 
start_time=time.time()
Wf, _,_ = jacobi_angles(R_tau_f)
Sf = np.dot(Wf.T,Xf)
print("--- {:.2f} seconds ---".format(time.time() - start_time))
#%%
plot_sources(Sf,Xf, flipped=True)
#%%

def plot_flips(source, sourcef):
    
    t = np.arange(0, (ts)/200, 1/200)
    f, axarr = plt.subplots(1,2, figsize=(12,1), sharex=True, sharey=True)
    axarr[0].set_title('source X')
    axarr[0].plot(t, source)
    axarr[1].set_title('source Xf')
    axarr[1].plot(t, sourcef)
    f.show()       
        
def find_flips(S,Sf):
    Sc = np.zeros((21,5601))
    i = 0
    for source in S:
        flip = False
        for sourcef in Sf:
            if(np.corrcoef(source,-sourcef)[0][1] > 0.99):
                plot_flips(source,sourcef)
                flip = True
                
        if(not flip):
            Sc[i,:] = source
        i = i+1
    return Sc


Sc = find_flips(S,Sf)

#%%
Xc = np.dot(W.T,Sc)
Vtc = np.dot(np.linalg.inv(U),Xc)
S=np.diag(s)
C = np.dot(np.dot(U, S), Vtc) 
plot_sources(Sc,C, corrected=True)

#%%
def plot_correction(X,C):
    
    
    return
