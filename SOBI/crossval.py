# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 10:44:01 2018

@author: Lisa
"""

import sys
sys.path.append('C:/Users/Lisa/Desktop/SOBI_TestData')

import PEEG_Analyse2 as pa
import SOBI as sobi
import Validate as val
from scipy import signal
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import rolling_corr
import time
import sim_data
Data = sim_data.SimData()
import csv
#%% init
diag=['Jac','Fro','ACDC','LSB']
#Sutherland(2004)
taus_std = np.array([0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,55,60,
                 65,70,75,80,85,90,95,100,120,140,160,180,200,220,240,260,280,
                 300])
taus_1 = np.array([0,1,2])
taus_2 = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
taus_3 = np.array([0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
taus_4 = np.array([25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,120,140,160,180,200,220,240,260,280,300])
#Taskinen(2016)
T1 = np.array([0,1])
T11 = np.arange(0,13)
T13 = np.arange(0,50)

taus = [taus_std,taus_1,taus_2,taus_3]#taus_4]#,T1,T11,T13]

epses = [0.1,0.01,0.001,0.0001]
#file='C:/Users/Lisa/Desktop/SOBI_TestData/'
#files = ['S0002O01M01_pEEG_CHDR1633_13OCT2017_125357.EDF','S0001O01M01_pEEG_CHDR1633_13OCT2017_092441.EDF',
#         'S0003O01M01_pEEG_CHDR1633_20OCT2017_132056.EDF','S0004O01M01_pEEG_CHDR1633_20OCT2017_095710.EDF',
#         'S0005O01M01_pEEG_CHDR1633_20OCT2017_125153.EDF','S0006O01M01_pEEG_CHDR1633_27OCT2017_095158.EDF',
#         'S0007O01M01_pEEG_CHDR1633_27OCT2017_124122.EDF','S0007O01M01_pEEG_CHDR1633_27OCT2017_131003.EDF',
#         'S0009O01M01_pEEG_CHDR1633_03NOV2017_121917.EDF','S0008O01M01_pEEG_CHDR1633_03NOV2017_093642.EDF']
subjects = 55
rhos = np.array([0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6])
inv = np.array([True,False])
timetable = np.zeros((55,4,4))
nmsetable = np.zeros((55,4,4))
tau_sets = ['t_sdt','t1','t2','t3']
#%%
nmse_X = np.zeros(55)
for f in range(1,subjects):
    print('Subject {}'.format(f))
    X = np.concatenate((Data.X['id{}'.format(f)],Data.HEOG['id{}'.format(f)], Data.VEOG['id{}'.format(f)]))
    
    sobi_J = sobi.SOBI(X, 
                     np.array([len(Data.electrodes), len(Data.electrodes)+1]),diag='Jac',taus=taus[2],eps=epses[2])
    Xc = np.zeros_like(X)
    Xc[:19] = sobi_J.Xc[:19]
    Xc[19:21] = X[19:21]
    validate = val.Validate(X,X,[19,20],B=Data.B['id{}'.format(f)])
    nmse = validate.NMSE()
    sumnmse = sum(abs(nmse))
    nmse_X[f-1] = sumnmse
#    with open('crossval/sim_{}_nmse.csv'.format(f), 'w', newline='') as csvfile:
#        fieldnames = ['channel','NMSE']
#        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#        writer.writeheader()
#        for i in range(len(nmse)):
#            writer.writerow({'channel':Data.electrodes[i] ,'NMSE':'{:.4f}'.format(nmse[i])})
#        writer.writerow({'channel':'\t' ,'NMSE':' '})
#        writer.writerow({'channel':'sum' ,'NMSE':'{:.4f}'.format(sumnmse)})
#       

#%%

for f in range(1,subjects):
    X = np.concatenate((Data.X['id{}'.format(f)],Data.HEOG['id{}'.format(f)], Data.VEOG['id{}'.format(f)]))
    
    print('Subject {}'.format(f))
    for d in range(len(diag)):
        for e in range(len(epses)):
            print(diag[d] + ', eps = {}'.format(epses[e]))
            with open('crossval/sim_{}_diag_{}_eps_0p{}1.csv'.format(f,diag[d],'0'*e), 'w', newline='') as csvfile:
                fieldnames = ['electrode', 'time','NMSE']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                start_time = time.time()
                sobi_J = sobi.SOBI(X,[19,20],diag=diag[d],taus=taus_3,eps=epses[e])
                Xc = np.zeros_like(X)
                Xc[:19] = sobi_J.Xc[:19]
                Xc[19:21] = X[19:21]

                t = time.time()-start_time
                print("--- {:.2f} seconds total ---".format(t))
                timetable[f,d,e] = t
                validate = val.Validate(X,Xc,[19,20],B=Data.B['id{}'.format(f)])
                nmse = validate.NMSE()
                sumnmse = sum(abs(nmse))
                nmsetable[f,d,e] = sumnmse
                for e in range(len(Data.electrodes)):
                    writer.writerow({'electrode':Data.electrodes[e], 'time':'   ' ,'NMSE':'{:.4f}'.format(nmse[e])})
                writer.writerow({'electrode':'sum', 'time':'{:.2f}'.format(t) ,'NMSE':'{:.4f}'.format(sumnmse)})
            ##bestand wegschrijven:
            #file -> taus,timetable[f,d,e],betatable[f,d,e,:,0],betatable[f,d,e,:,1]
            #filename = 'subject_{f}_diag_{diag[d]}_eps_{epses[e]}'

#%%

nmse_mean = np.mean(nmsetable,axis = 0)
time_mean = np.mean(timetable,axis = 0)

for d in range(len(diag)):
    for e in range(len(epses)):
        print(diag[d] + ', eps = {}'.format(epses[e]))
        with open('crossval/sim_mean_diag_{}_eps_0p{}1.csv'.format(diag[d],'0'*e), 'w', newline='') as csvfile:
            fieldnames = ['tau', 'time','NMSE']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            t= time_mean[d,e,tau]
            nmse = nmse_mean[d,e,0] 
            writer.writerow({'tau':tau_sets[tau] , 'time':'{:.2f}'.format(t) ,'NMSE':'{:.4f}'.format(nmse)})
#%%
nmse_mean_eps = np.mean(nmse_mean,axis=1)
time_mean_eps = np.mean(time_mean,axis=1)
for d in range(len(diag)):
    print(diag[d])
    with open('crossval/subject_mean_diag_{}_eps_mean.csv'.format(diag[d],'0'*e), 'w', newline='') as csvfile:
        fieldnames = ['tau', 'time','NMSE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for tau in range(len(taus)):
            t= time_mean_eps[d,tau]
            nmse = nmse_mean_eps[d,tau] 
            writer.writerow({'tau':tau_sets[tau] , 'time':'{:.2f}'.format(t) ,'NMSE':'{:.4f}'.format(sumnmse)})

#%% Diag = Jac, epses= 0.0001, tau = ??

timetable = np.zeros((55,11,2))
nmsetable = np.zeros((55,11,2))     
for f in range(1,subjects):
    X = np.concatenate((Data.X['id{}'.format(f)],Data.HEOG['id{}'.format(f)], Data.VEOG['id{}'.format(f)]))
    
    print('Subject {}'.format(f))
    for r in range(len(rhos)):
        for i in range(len(inv)):
            print('rho {}, inv {}'.format(rhos[r],inv[i]))
            with open('crossval/subject_{}_rho_{}_inv_{}.csv'.format(f,rhos[r],inv[i]), 'w', newline='') as csvfile:
                fieldnames = ['electrode', 'time','NMSE']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                start_time = time.time()
                sobi_J = sobi.SOBI(X,[19,20],diag='Jac',taus=taus_3,eps=0.01, inversion=inv[i], corr_thres=r)
                Xc = np.zeros_like(X)
                Xc[:19] = sobi_J.Xc[:19]
                Xc[19:21] = X[19:21]

                t = time.time()-start_time
                print("--- {:.2f} seconds total ---".format(t))
                timetable[f,r,i] = t
                validate = val.Validate(X,Xc,[19,20],B=Data.B['id{}'.format(f)])
                nmse = validate.NMSE()
                sumnmse = sum(abs(nmse))
                nmsetable[f,r,i] = sumnmse
                for e in range(len(Data.electrodes)):
                    writer.writerow({'electrode':Data.electrodes[e], 'time':'   ' ,'NMSE':'{:.4f}'.format(nmse[e])})
                writer.writerow({'electrode':'sum', 'time':'{:.2f}'.format(t) ,'NMSE':'{:.4f}'.format(sumnmse)})
#%%
nmse_mean = np.mean(nmsetable,axis = 0)
time_mean = np.mean(timetable,axis = 0)

for r in range(len(rhos)):
    for i in range(len(inv)):
        print('rho = {}, inv = {}'.format(rhos[r],inv[i]))
        with open('crossval/subject_mean_rho_{}_inv_{}.csv'.format(rhos[r],inv[i]), 'w', newline='') as csvfile:
            fieldnames = ['time','NMSE']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            t= time_mean[r,i]
            nmse = nmse_mean[r,i] 
            writer.writerow({'time':'{:.2f}'.format(t) ,'NMSE':'{:.4f}'.format(nmse)})