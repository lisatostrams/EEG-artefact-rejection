# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:39:17 2017

@author: Lisa Tostrams
"""

import scipy
from scipy import signal
import numpy as np
from sim_data import SimData
from joint_diagonalizer import jacobi_angles
import time

Data = SimData()

import matplotlib.pyplot as plt
saveplots = True
id = 8
#%%
X = np.concatenate((Data.X['id{}'.format(id)],Data.HEOG['id{}'.format(id)], Data.VEOG['id{}'.format(id)]))
nrlags = 500
ts = len(X.T)
#taus = [int(x) for x in scipy.stats.truncnorm.rvs((0 - ts) / (ts/4), ((ts*2)-1 - ts) / (ts/4), loc=ts, scale=ts/4, size=nrlags)]

'''
Important: choose the right lags! 
The most important limitation on computation time and performance. 
'''
taus = np.concatenate((np.arange(-100,100,1))) 
taus = [t+5600 for t in taus]

#%%
def svd_whiten(X):
    '''
    Whiten input matrix X with singular value decomposition
    both U and Vt are orthonormal matrices, their multiplication results in whitened matrix
    '''
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    X_white = np.dot(U, Vt)
    return X_white, U, s, Vt



def cross_corr(X):
    '''
    Cross correlation function: calculates the convolution (correlation in whitened data) matrix for all lags from -len(signal) to +len(signal)
    '''
    n_signals = len(X)
    ts = len(X.T)
    OUT = np.zeros([(2*ts)-1,n_signals,n_signals])
    for first in range(n_signals):
        for second in range(n_signals):
            OUT[:,second,first]= signal.fftconvolve(X[first], X[second], mode='full')  

    return OUT
    
    
X, U, s, Vt = svd_whiten(X)
s=np.diag(s)
R_tau = cross_corr(X)
'''
Choose the correlations at predetermined lags
'''
R_tau_ = R_tau[taus] 

#%%
'''
Calculate the joint diagonalizer for all correlation matrices using method with jacobi angles
The transpose of the joint diagonalizer is the unmixing matrix
Computation time is a function of number of lags
'''
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
        f.savefig('Figures/Sources/X_S_{}.png'.format(id),dpi=300)
        #plt.close('all')
    if(save and flipped):
        f.savefig('Figures/Sources/X_S_{}_flippedeog.png'.format(id),dpi=300)
    if(save and corrected):
        f.savefig('Figures/Sources/C_S_{}_corrected.png'.format(id),dpi=300)
        
        
    f.show()

#%%
'''
Plot the sources determined by unmixing matrix
'''
plot_sources(S, X, save=saveplots)

#%%
'''
FOR AUTOMATED SOBI:
    Repeat the procedure with the EOG channels inverted
    
'''
Xf = np.concatenate((Data.X['id{}'.format(id)],-Data.HEOG['id{}'.format(id)], -Data.VEOG['id{}'.format(id)]))
Xf,_,_,_ = svd_whiten(Xf)
R_tauf = cross_corr(Xf)
R_tau_f = R_tauf[taus] 
start_time=time.time()
Wf, _,_ = jacobi_angles(R_tau_f)
Sf = np.dot(Wf.T,Xf)
print("--- {:.2f} seconds ---".format(time.time() - start_time))
#%%
plot_sources(Sf,Xf, save=saveplots, flipped=True)
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
    '''
    Find the sources that flipped (inverted) compared to previous sources
    '''
    Sc = np.zeros((21,ts))
    i = 0
    for source in S:
        flip = False
        for sourcef in Sf:
            if(np.corrcoef(source,-sourcef)[0][1] > 0.95):
                plot_flips(source,sourcef)
                flip = True
                
        if(not flip):
            Sc[i,:] = source
        i = i+1
    return Sc


    
def find_components_originating_near_eyes(S,W):
    '''
    Find the sources that originate near the eyes
    '''
    idH = len(Data.electrodes)
    idV = idH+1
    eye_comps = W[idH,:] > 0.3
    Sc = np.zeros((21,ts))
    Sc[:,:] = S[:,:]
    Sc[eye_comps,:] = 0
    eye_comps = W[idH,:] < -0.3
    Sc[eye_comps,:] = 0
    eye_comps = W[idV,:] > 0.3
    Sc[eye_comps,:]=0
    eye_comps = W[idV,:] < -0.3
    Sc[eye_comps,:] = 0
    return Sc, eye_comps
    
    
'''
AUTOMATED SOBI:
    find the sources that flipped compared to previous sources
    mark these as artefact sources and do not include them in reconstruction
    
    optional: find the components that originate near the eyes
    use geometric information from the Mixing matrix:
        which sources make up the EOG channels?
'''
Sc = find_flips(S,Sf)
#Sc, eyecomps = find_components_originating_near_eyes(Sc,W)



#%%
'''
Reconstruct the data from the new set of sources
'''
Xc = np.dot(W,Sc)
'''
Unwhiten the data
'''
Vtc = np.dot(np.linalg.inv(U),Xc)
C = np.dot(np.dot(U, s), Vtc) 
plot_sources(Sc,Xc, save=saveplots,corrected=True)

#%%
def plot_correction(X,C, save=False):

    f, axarr = plt.subplots(21, 3, figsize=(18,30),sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.05, hspace = 0.3)
    t = np.arange(0, (len(X[0]-1))/200, 1/200) 
    axarr[0,0].set_title('$B^s(t)$')
    axarr[0,1].set_title('$X^s(t)$')
    axarr[0,2].set_title('$C^s(t)$')

    B = Data.B['id{}'.format(id)]
    for i in range(19):
        axarr[i,0].set_ylabel(Data.electrodes[i])
        axarr[i,0].plot(t, B[i])
        axarr[i,1].plot(t, X[i])
        axarr[i,2].plot(t, C[i])
    axarr[19,0].axis('off')
    axarr[20,0].axis('off')
   
    axarr[19,1].set_ylabel('HEOG')
    axarr[19,1].plot(t, X[19])
    axarr[19,2].plot(t, C[19])
    axarr[20,1].set_ylabel('VEOG')
    axarr[20,1].plot(t, X[20])
    axarr[20,2].plot(t, C[20])
    f.tight_layout()
    if(save):
        f.savefig('Figures/Corrections/B_X_C_{}.png'.format(id),dpi=300)
        
    f.show()
X = np.concatenate((Data.X['id{}'.format(id)],Data.HEOG['id{}'.format(id)], Data.VEOG['id{}'.format(id)]))    
plot_correction(X,C, save=saveplots)

