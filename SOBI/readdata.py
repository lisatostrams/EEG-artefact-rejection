# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:02:03 2018

@author: Lisa
"""
import csv
import numpy as np
rhos = np.array([0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6])
inv = np.array([True,False])
timetable = np.zeros((10,11,2,4))
betatable = np.zeros((10,11,2,4,2))
tau_sets = ['t_sdt','t1','t2','t3']

for f in range(10):
    for r in range(len(rhos)):
        for i in range(len(inv)):        
            with open('crossval/subject_{}_rho_{}_inv_{}.csv'.format(f+1,rhos[r],inv[i]), 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                fieldnames = ['tau', 'time','beta_V','beta_H']
                tau=0
                for row in reader:
                    timetable[f,r,i,tau] = row['time']
                    betatable[f,r,i,tau,1] = row['beta_H']
                    betatable[f,r,i,tau,0] = row['beta_V']
                    tau=tau+1
                    