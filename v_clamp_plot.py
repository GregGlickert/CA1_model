import h5py
import matplotlib.pyplot as plt
import numpy as np

A = h5py.File('output/se_clamp_report.h5', 'r')
B = h5py.File('output/se_clamp_report1.h5', 'r')
C = h5py.File('output/se_clamp_report2.h5', 'r')
D = h5py.File('output/se_clamp_report3.h5', 'r')
E = h5py.File('output/se_clamp_report4.h5', 'r')

report = A['data'][1000:]
report1 = B['data'][1000:]
report2 = C['data'][1000:]
report3 = D['data'][1000:]
report4 = E['data'][1000:]

plt.plot(report, color='gray')
plt.plot(report1, color='gray')
plt.plot(report2, color='gray')
plt.plot(report3, color='gray')
plt.plot(report4, color='gray')
avg = np.mean(np.array([report, report1, report2, report3, report4]), axis=0)
plt.plot(avg, color='blue')
plt.ylabel("current")
plt.xlabel("time")
plt.title("Voltage clamp on AAC at -70mV")
plt.show()
