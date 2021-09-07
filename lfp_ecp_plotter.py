import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from mpl_toolkits.mplot3d import Axes3D
import h5py
import matplotlib.pyplot as plt
from scipy.signal import welch
import scipy.signal as ss

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def zscore(x):
    return (x - np.mean(x))/np.std(x)


tsim = 1000

lfp_file = "output1/ecp.h5"

f = h5py.File(lfp_file,'r')
lfp = list(f['ecp']['data'])
lfp_arr = np.asarray(lfp)

plt.figure()
plt.title('LFP')
lfp1 = zscore(lfp_arr[:,0])
#lfp2 = zscore(lfp_arr[:,2])
#lfp3 = zscore(lfp_arr[:,4])
#np.savetxt("lfp_ben.csv", lfp1, delimiter=",")
plt.plot(np.arange(0,tsim,0.1),lfp1)
#plt.plot(np.arange(0,tsim,0.1),lfp2)
#plt.plot(np.arange(0,tsim,0.1),lfp3)
plt.xlim(4000,5000)

freqs, psd = welch(lfp1,fs=1000)


plt.figure()
plt.semilogy(freqs, psd)
plt.title('LFP PSD')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.show()


lfp_file = "output1/ecp.h5"

f = h5py.File(lfp_file,'r')
lfp = list(f['ecp']['data'])
lfp_arr = np.asarray(lfp)
plt.plot(np.arange(0,tsim,0.1),lfp_arr)
plt.show()

