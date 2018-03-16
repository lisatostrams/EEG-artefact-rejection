# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:18:08 2018

@author: Lisa
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA

# #############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time+1.2)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
#s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
s3 = np.sin(2 * time)  # Signal 1 : sinusoidal signal

S = np.c_[s1, s2, s3]
noise =  0.5 * np.random.normal(size=S.shape)
S[:,0] += np.mean(noise,axis=1)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# #############################################################################
# Plot results
#%%
plt.figure(figsize=(10,3))
#runfile('C:/Users/Lisa/Desktop/EEG-artefact-rejection/SOBI/SOBI.py', wdir='C:/Users/Lisa/Desktop/EEG-artefact-rejection/SOBI')
#sobi = SOBI(X.T, [])
#sobi.S
#sS = sobi.S
models = [X, np.array([S[:,0],S[:,1],S[:,2],np.mean(noise,axis=1)]).T, S_, sS.T]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'SOBI recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(2, 2, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

#plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.tight_layout()

plt.savefig('icavssobi_lagged.png', dpi=300)
