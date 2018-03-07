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
file='C:/Users/Lisa/Desktop/SOBI_TestData/'
files = ['S0002O01M01_pEEG_CHDR1633_13OCT2017_125357.EDF','S0001O01M01_pEEG_CHDR1633_13OCT2017_092441.EDF',
         'S0003O01M01_pEEG_CHDR1633_20OCT2017_132056.EDF','S0004O01M01_pEEG_CHDR1633_20OCT2017_095710.EDF',
         'S0005O01M01_pEEG_CHDR1633_20OCT2017_125153.EDF','S0006O01M01_pEEG_CHDR1633_27OCT2017_095158.EDF',
         'S0007O01M01_pEEG_CHDR1633_27OCT2017_124122.EDF','S0007O01M01_pEEG_CHDR1633_27OCT2017_131003.EDF',
         'S0009O01M01_pEEG_CHDR1633_03NOV2017_121917.EDF','S0008O01M01_pEEG_CHDR1633_03NOV2017_093642.EDF']

rhos = np.array([0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6])
inv = np.array([True,False])
timetable = np.zeros((10,4,4,4))
betatable = np.zeros((10,4,4,4,2))
tau_sets = ['t_sdt','t1','t2','t3']
#%%
for f in range(10,len(files)):
    peeg = pa.PEEG_Analyse2(file + files[f])
    X = peeg.readSignals()
    print('Subject {}'.format(f+1))
    sobi_J = sobi.SOBI(X[0:23],[21,22],diag='Jac',taus=taus[2],eps=epses[2])
    Xc = np.zeros_like(X)
    Xc[:21] = sobi_J.Xc[:21]
    Xc[21:24] = X[21:24]
    validate = val.Validate(X,Xc,[21,22],peeg=peeg)
    bX,bXc = validate.regression()
    beta = sum(abs(bX))
    with open('crossval/subject_{}_beta.csv'.format(f+1), 'w', newline='') as csvfile:
        fieldnames = ['channel','beta_V','beta_H']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(bX)):
            writer.writerow({'channel':peeg.signalLabels[i] ,'beta_V':'{:.4f}'.format(bX[i,0]),
                                     'beta_H':'{:.4f}'.format(bX[i,1])})
        writer.writerow({'channel':'\t' ,'beta_V':' ',
                                     'beta_H':' '})
        writer.writerow({'channel':'sum abs' ,'beta_V':'{:.4f}'.format(beta[0]),
                                     'beta_H':'{:.4f}'.format(beta[1])})
       

#%%

for f in range(2,10):
    peeg = pa.PEEG_Analyse2(file + files[f])
    X = peeg.readSignals()
    print('Subject {}'.format(f+1))
    for d in range(len(diag)):
        for e in range(len(epses)):
            print(diag[d] + ', eps = {}'.format(epses[e]))
            with open('crossval/subject_{}_diag_{}_eps_0p{}1.csv'.format(f+1,diag[d],'0'*e), 'w', newline='') as csvfile:
                fieldnames = ['tau', 'time','beta_V','beta_H']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for tau in range(len(taus)):
                    start_time = time.time()
                    sobi_J = sobi.SOBI(X[0:23],[21,22],diag=diag[d],taus=taus[tau],eps=epses[e])
                    Xc = np.zeros_like(X)
                    Xc[:21] = sobi_J.Xc[:21]
                    Xc[21:24] = X[21:24]
                    t = time.time()-start_time
                    print("--- {:.2f} seconds total ---".format(t))
                    timetable[f,d,e,tau] = t
                    validate = val.Validate(X,Xc,[21,22],peeg=peeg)
                    bX,bXc = validate.regression()
                    beta = sum(abs(bXc))
                    betatable[f,d,e,tau] = beta
                    writer.writerow({'tau':tau_sets[tau] , 'time':'{:.2f}'.format(t) ,'beta_V':'{:.4f}'.format(beta[0]),
                                     'beta_H':'{:.4f}'.format(beta[1])})
            ##bestand wegschrijven:
            #file -> taus,timetable[f,d,e],betatable[f,d,e,:,0],betatable[f,d,e,:,1]
            #filename = 'subject_{f}_diag_{diag[d]}_eps_{epses[e]}'

#%%
beta_mean = np.mean(betatable,axis = 0)
time_mean = np.mean(timetable,axis = 0)

for d in range(len(diag)):
    for e in range(len(epses)):
        print(diag[d] + ', eps = {}'.format(epses[e]))
        with open('crossval/subject_mean_diag_{}_eps_0p{}1.csv'.format(diag[d],'0'*e), 'w', newline='') as csvfile:
            fieldnames = ['tau', 'time','beta_V','beta_H']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for tau in range(len(taus)):
                t= time_mean[d,e,tau]
                beta = beta_mean[d,e,tau] 
                writer.writerow({'tau':tau_sets[tau] , 'time':'{:.2f}'.format(t) ,'beta_V':'{:.4f}'.format(beta[0]),
                                 'beta_H':'{:.4f}'.format(beta[1])})
#%%
beta_mean_eps = np.mean(beta_mean,axis=1)
time_mean_eps = np.mean(time_mean,axis=1)
for d in range(len(diag)):
    print(diag[d])
    with open('crossval/subject_mean_diag_{}_eps_mean.csv'.format(diag[d],'0'*e), 'w', newline='') as csvfile:
        fieldnames = ['tau', 'time','beta_V','beta_H']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for tau in range(len(taus)):
            t= time_mean_eps[d,tau]
            beta = beta_mean_eps[d,tau] 
            writer.writerow({'tau':tau_sets[tau] , 'time':'{:.2f}'.format(t) ,'beta_V':'{:.4f}'.format(beta[0]),
                             'beta_H':'{:.4f}'.format(beta[1])})

#%% Diag = Jac, epses= 0.0001, tau = ??
            
for f in range(3,len(files)):
    peeg = pa.PEEG_Analyse2(file + files[f])
    X = peeg.readSignals()
    print('Subject {}'.format(f+1))
    for r in range(len(rhos)):
        for i in range(len(inv)):
            print('rho {}, inv {}'.format(rhos[r],inv[i]))
            with open('crossval/subject_{}_rho_{}_inv_{}.csv'.format(f+1,rhos[r],inv[i]), 'w', newline='') as csvfile:
                fieldnames = ['tau', 'time','beta_V','beta_H']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for tau in range(len(taus)):
                    start_time = time.time()
                    sobi_J = sobi.SOBI(X[0:23],[21,22],diag='Jac',taus=taus[tau],eps=0.0001, corr_thres=rhos[r], inversion=inv[i])
                    Xc = np.zeros_like(X)
                    Xc[:21] = sobi_J.Xc[:21]
                    Xc[21:24] = X[21:24]
                    t = time.time()-start_time
                    print("--- {:.2f} seconds total ---".format(t))
                    timetable[f,d,e,tau] = t
                    validate = val.Validate(X,Xc,[21,22],peeg=peeg)
                    bX,bXc = validate.regression()
                    beta = sum(abs(bXc))
                    betatable[f,d,e,tau] = beta
                    writer.writerow({'tau':tau_sets[tau] , 'time':'{:.2f}'.format(t) ,'beta_V':'{:.4f}'.format(beta[0]),
                                     'beta_H':'{:.4f}'.format(beta[1])})