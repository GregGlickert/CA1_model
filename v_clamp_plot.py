import h5py
import matplotlib.pyplot as plt
import numpy as np

F = h5py.File('output/se_clamp_report2.h5', 'r')
report = F['data'][2000:]
x = np.arange(0, 100, 0.1)
plt.plot(x, report)
plt.ylabel("current (nA)")
plt.xlabel("time (ms)")
#plt.xlim((200,300))
plt.title("Voltage clamp on AAC")
plt.show()