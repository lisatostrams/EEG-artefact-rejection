# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:14:17 2017

@author: Lisa Tostrams
"""
import sim_data
import numpy as np
Data = sim_data.SimData()
id=5
#Data.plot_subject(id)
#%%
import SOBI
Sobi = SOBI.SOBI(np.concatenate((Data.X['id{}'.format(id)],Data.HEOG['id{}'.format(id)], Data.VEOG['id{}'.format(id)])),np.array([len(Data.electrodes), len(Data.electrodes)+1]), eps = 1e-2, sweeps=500)
#epses = [0.1,0.01, 1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
#runtimes = np.zeros([8,1])
#NMSEsum = np.zeros([8,1])
#i=0
#for e in epses:
#    start_time=time.time()
#    Sobi = SOBI.SOBI(np.concatenate((Data.X['id{}'.format(id)],Data.HEOG['id{}'.format(id)], Data.VEOG['id{}'.format(id)])), 
#                 np.array([len(Data.electrodes), len(Data.electrodes)+1]),
#                 eps=e)
#    runtimes[i] = time.time() - start_time
#    NMSEsum[i] = sum(NMSE(Sobi.Xc))
#    i=i+1
#%%
plt.plot(epses,runtimes,'-o',label='Runtimes')
plt.plot(epses, NMSEsum,'-o',label='Summed NMSE')
#plt.semilogx(basex=e)
plt.semilogy()
plt.semilogx()
#plt.ylim([0,12])
plt.legend()
plt.show()
#%%
import matplotlib.pyplot as plt
labels = Data.electrodes + ['HEOG','VEOG']
plot_correction(Sobi.X,Sobi.Xc)
#%%
print('NMSE sum = {}'.format(sum(NMSE(Sobi.Xc))))

#%%
def plot_correction(X,C, save=False):
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
        f.savefig('Figures/Corrections/B_X_C_{}.png'.format(id),dpi=300)
        
    f.show()
    
    
def NMSE(C):
    B = Data.B['id{}'.format(id)]
    Ct = C[0:19,:]
    return sum(((Ct-B)**2).T)/sum((B**2).T)
    
#%%
import matplotlib.transforms as mtransforms
def annotate(B,X):
    diff = abs(X-B)
    OA = diff > 5
    return OA

def show_OA(B,X,OA):
    f, axarr = plt.subplots(21, 2, figsize=(12,30),sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.05, hspace = 0.3)
    t = np.arange(0, (len(Data.X['id{}'.format(id)][0]-1))/200, 1/200) 
    axarr[0,0].set_title('$B^s(t)$')
    axarr[0,1].set_title('$X^s(t)$')
    for i in range(19):
        axarr[i,0].set_ylabel(Data.electrodes[i])
        axarr[i,0].plot(t, B[i])
        axarr[i,1].plot(t, X[i])
        trans = mtransforms.blended_transform_factory(axarr[i,1].transData, axarr[i,1].transAxes)
        axarr[i,1].fill_between(t, 0, 1, where=OA[i,:], facecolor='red', alpha=0.5, transform=trans)
        
    axarr[19,0].set_title('$O^s(t)$')
    axarr[19,0].set_ylabel('HEOG')
    axarr[19,0].plot(t, Data.HEOG['id{}'.format(id)][0])
    axarr[19,1].axis('off')
    axarr[20,0].set_ylabel('VEOG')
    axarr[20,0].plot(t, Data.VEOG['id{}'.format(id)][0])
    axarr[20,1].axis('off')
    f.tight_layout()

    
B = Data.B['id{}'.format(id)]
X = Data.X['id{}'.format(id)]
C = Sobi.Xc[0:19,:]
OA = annotate(C,X)
show_OA(C,X,OA)

#%%
    
    
    
    