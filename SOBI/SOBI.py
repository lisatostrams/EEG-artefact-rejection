# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:51:34 2017

@author: RjDoll & Lisa Tostrams

"""
import numpy as np
from scipy import signal
from joint_diagonalizer import jacobi_angles, fast_frobenius,ACDC,LSB
import time
import matplotlib.pyplot as plt


class SOBI(object):

    def __init__(self, X, EOG_chans, taus = np.array([0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,55,60,
                 65,70,75,80,85,90,95,100,120,140,160,180,200,220,240,260,280,
                 300]), corr_thres=.3, eps = 1e-3, sweeps = 500, diag='Jac'):
        """
        Constructor input: X, EOG_channels, optional: taus, corr_thres, eps, sweeps
        
        
        """
        self.X = X
        ts = len(X.T)
        self.taus = taus+ts-1
        self.EOG_chans = EOG_chans
        self.corr_thres = corr_thres
        ## Whiten data using SVD, save the variables required for unwhitening
        self.X_white, U, s, Vt = self.svd_whiten(self.X)
        s=np.diag(s)
        
        ## Calculate cross-correlations: (1) for each time-lag, (2) extract those at taus
        self.R_tau = self.cross_corr(self.X_white)
        print(len(self.R_tau))
        self.R_tau = self.R_tau[self.taus] 
        
        ## Joint-diagonalisation
        self.S, self.W = self.joint_diag(self.X_white, self.R_tau, diag, eps, sweeps)
        
        ## Flip EOG channels. For now: assume last two sensors contain EOG
        self.X_flipped = np.copy(self.X_white)
        self.X_flipped[self.EOG_chans,:] = -self.X_flipped[self.EOG_chans,:]
        R_tauf = self.cross_corr(self.X_flipped)
        R_tauf = R_tauf[taus] 
        # Joint-diagonalisation
        self.Sf, Wf = self.joint_diag(self.X_flipped, R_tauf, eps, sweeps)

        # Find flips
        self.Sc = self.find_flips(self.S, self.Sf)
        

        #% Find components which correlate highly with EOG channels
        for i in range(0, len(self.EOG_chans)):
            for j in range(0, len(self.Sc)):
                coef = np.corrcoef(self.Sc[j,:], self.X[self.EOG_chans[i]])[0,1]
                if np.abs(coef) > self.corr_thres:
                    self.Sc[j,:] = np.zeros([1,self.Sc.shape[1]])
        #% Reconstruct
        self.Xc = self.reconstruct(self.Sc, self.W, U, s)
                
        
    
    def plot_correction(self,X,Xc,labels,beg,Ts):
        N = X.shape[1]
        f, axarr = plt.subplots(len(X), 2, figsize=(8,20),sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.05, hspace = 0.3)
        t = np.arange(beg*Ts, beg*Ts + N*Ts, Ts) #np.arange(0, (len(X[0]-1))/200, 1/200) 
        axarr[0,0].set_title('Recorded')# $X^s(t)$')
        axarr[0,1].set_title('Corrected')#$C^s(t)$')
        for i in range(0,len(X)):
            axarr[i,0].plot(t,X[i])
            axarr[i,0].set_ylabel(labels[i])
            axarr[i,1].plot(t,Xc[i])
        axarr[i,0].set_xlabel('Time [s]')
        axarr[i,1].set_xlabel('Time [s]')
        f.tight_layout()
        f.show()
        
    def reconstruct(self, Sc, W, U, s):
        '''
        Reconstruct the data from the new set of sources
        Unwhiten the data
        '''
        tmp = np.dot(W, Sc)
        Vtc= np.dot(np.linalg.inv(U),tmp)
        return np.dot(np.dot(U, s), Vtc)
        
    def svd_whiten(self, X):
        '''
        Whiten input matrix X with singular value decomposition
        both U and Vt are orthonormal matrices, their multiplication results in whitened matrix
        '''
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        X_white = np.dot(U, Vt)
        return X_white, U, s, Vt


    def cross_corr(self, X):
        '''
        Cross correlation function: calculates the convolution (correlation in whitened data) matrix for all lags from -len(signal) to +len(signal)
        '''
        n_signals = len(X)
        ts = len(X.T)
        OUT = np.zeros([(2*ts)-1,n_signals,n_signals])
        for first in range(n_signals):
            for second in range(first+1):
                OUT[:,second,first]= signal.fftconvolve(X[first], X[second][::-1], mode='full')  
                OUT[:,first,second] = OUT[:,second,first]
        return OUT
        
    def joint_diag(self,X, R_tau, diag, eps = 1e-3, sweeps = 500):
        '''
        Calculate the joint diagonalizer for all correlation matrices
        The transpose of the joint diagonalizer is the unmixing matrix
        Computation time is a function of number of lags
        '''
        start_time = time.time()
        if(diag == 'ACDC'):
            W,_,_,_ = self.svd_whiten(ACDC(R_tau, eps=eps, sweeps=sweeps))            
        elif(diag == 'Fro'):
            W,_ = fast_frobenius(R_tau, eps = eps)   
            W,_,_,_ = self.svd_whiten(W)
        elif(diag == 'LSB'):
           W,_,_,_ = self.svd_whiten(LSB(R_tau))
        else:
            W,_,_ = jacobi_angles(R_tau, eps = eps, sweeps = sweeps)
   
        S = np.dot(W.T,X)
        print("--- {:.2f} seconds ---".format(time.time() - start_time))
        return S, W
        
    def find_flips(self,S,Sf):
        '''
        Find the sources that flipped (inverted) compared to previous sources
        '''
        Sc = np.zeros((S.shape[0],S.shape[1]))
        i = 0
        for source in S:
            flip = False
            for sourcef in Sf:
                coef = np.corrcoef(source,-sourcef)[0][1]
                if(coef > 0.95):
                    flip = True                
            if(not flip):
                Sc[i,:] = source
            i = i+1
        return Sc