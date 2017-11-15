# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:23:41 2017

@author: Lima
"""

import sim_data
import SOBI
import matplotlib.pyplot as plt
import time
Data = sim_data.SimData()
import numpy as np

def NMSE(C,id):
    B = Data.B['id{}'.format(id)]
    Ct = C[0:19,:]
    return sum(((Ct-B)**2).T)/sum((B**2).T)
    

##%%
#epses = [0.1,0.01, 1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]  
#runtimes = np.zeros([54,8])
#NMSEsum = np.zeros([54,8])
#for id in range(1,55):
#    i=0
#    for e in epses:
#        start_time=time.time()
#        Sobi = SOBI.SOBI(np.concatenate((Data.X['id{}'.format(id)],Data.HEOG['id{}'.format(id)], Data.VEOG['id{}'.format(id)])), 
#                     np.array([len(Data.electrodes), len(Data.electrodes)+1]),
#                     eps=e)
#        runtimes[id-1,i] = time.time() - start_time
#        NMSEsum[id-1,i] = sum(NMSE(Sobi.Xc,id))
#        i=i+1
#
#    plt.plot(epses,runtimes[id-1,:],'-o',label='Runtimes')
#    plt.plot(epses, NMSEsum[id-1,:],'-o',label='Summed NMSE')
#    #plt.semilogx(basex=e)
#    plt.semilogy()
#    plt.semilogx()
#    #plt.ylim([0,12])
#    plt.legend()
#    plt.savefig('Figures/id{}_NMSEvsRuntime_jacobi_diagonalizer.png'.format(id))
#    plt.close()
#
#    
#    
##%%
#plt.plot(epses,np.mean(runtimes,axis=0),'-o', label='Runtimes')
#plt.plot(epses,np.mean(NMSEsum,axis=0),'-o', label='Summed NMSE')
#plt.xlabel('eps')
#plt.semilogx()
#plt.semilogy()
#plt.legend()
#plt.savefig('Figures/Average_NMSEvsRuntime_Jacobi_diagonalizer.png',dpi=300)
##%%
#plt.plot(epses,np.mean(runtimes,axis=0),'-o', label='Runtime')
#plt.grid()
#plt.semilogx()
#plt.legend()
#plt.savefig('Figures/Average_Runtime_Jacobi_diagonalizer.png',dpi=300)
#%%
import gc
def plot_correction(X,C,id, save=False,diag='Jac'):
    f, axarr = plt.subplots(21, 3, figsize=(12,20),sharex=True, sharey=True)
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
        f.savefig('Figures/Corrections/B_X_C_{}_{}.png'.format(id,diag),dpi=300)
    f.clf()
    plt.close(f)
    plt.close()
        


    

runtimes_J = np.zeros([54,1])
runtimes_F = np.zeros([54,1])

NMSEsum_J = np.zeros([54,1])
NMSEsum_F = np.zeros([54,1])

for id in range(1,55):
    print(id)
    print("J")
    start_time=time.time()
    Sobi = SOBI.SOBI(np.concatenate((Data.X['id{}'.format(id)],Data.HEOG['id{}'.format(id)], Data.VEOG['id{}'.format(id)])), 
                     np.array([len(Data.electrodes), len(Data.electrodes)+1]),
                     eps=1e-2, diag = 'Jac')
    runtimes_J[id-1] = time.time() - start_time
    NMSEsum_J[id-1] = sum(NMSE(Sobi.Xc,id))
  #  plot_correction(Sobi.X,Sobi.Xc,id,save=True)
    print("F")
    start_time=time.time()
    Sobi = SOBI.SOBI(np.concatenate((Data.X['id{}'.format(id)],Data.HEOG['id{}'.format(id)], Data.VEOG['id{}'.format(id)])), 
                 np.array([len(Data.electrodes), len(Data.electrodes)+1]),
                 diag = 'Fro', sweeps=1000)
    runtimes_F[id-1] = time.time() - start_time
    NMSEsum_F[id-1] = sum(NMSE(Sobi.Xc,id))
  #  plot_correction(Sobi.X, Sobi.Xc, id, save=True, diag='Fro')
#%%
plt.plot(runtimes_J,runtimes_F,'.')
#%%
plt.plot(NMSEsum_J,NMSEsum_F,'.')
    