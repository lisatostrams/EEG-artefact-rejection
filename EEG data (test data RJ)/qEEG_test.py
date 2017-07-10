# -*- coding: utf-8 -*-
import pyedflib
import numpy as np
import matplotlib.pyplot as plt

#filename = "D:\\Python_workspace\CHDR1703_EDF_test\\data\\S9999O99M99_EyesCOn_CHDRXXXX_03JUL2017_115127.EDF"
#filename = "D:\\Python_workspace\CHDR1703_EDF_test\\data\\SXXXXOXXMXX_EyesCO_CHDRXXXX_03JUL2017_172750.EDF"
#filename = "D:\\Python_workspace\CHDR1703_EDF_test\\data\\S1234O01M01_EyesClosedEyesOpen_CHDRXXXX_30JUN2017_144440.edf"
filename = "C:\\Users\\Lima\\Desktop\\CHDR\\EEG data (test data RJ)\\S1111O01M01_EyesCO_CHDRXXXX_06JUL2017_151733.edf"
f = pyedflib.EdfReader(filename)

n = f.signals_in_file
signal_labels = f.getSignalLabels()

fs = f.getSampleFrequency(0)
Ts = 1/fs
nSamples = f.getNSamples()[0]
T  = nSamples * Ts
tt = np.linspace(Ts, T, nSamples)

Cz = f.readSignal(signal_labels.index('Cz'))
Fz = f.readSignal(signal_labels.index('Fp1'))
EOG = f.readSignal(signal_labels.index('EOG'))

trigger = f.readSignal(signal_labels.index('Trigger_abs'))


# return triggers
def find_triggers(channel, value):
    return np.array(np.where(np.diff(channel) == value)).flatten()+1

# Kanalen/afleidingen of interest definieren 

# Opsplitsen in blokken van 2^x samples (2 - 8 sec). Overweeg overlap
# window in seconds
# Overlap between 0 and 1
# start_pos and end_pos are indices
def split_into_epochs(channel, window, overlap, start_pos, end_pos): 
    #window = window * fs # Change window from seconds to samples
    #l_between = end_pos - start_pos
    #n_epochs = 
    
    pass
# Check voor artefacten in het EOG - reject epoch

# Bereken PSD per Epoch

# Middel(?) PSDs

# Opsplitsen in banden

# AUC bepalen per band per kanaal/afleiding
start_clsd = find_triggers(trigger, 1)
end_clsd   = find_triggers(trigger, 2)

start_opnd = find_triggers(trigger, 4)
end_opnd   = find_triggers(trigger, 8)
