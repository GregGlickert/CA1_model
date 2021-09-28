import h5py
import matplotlib.pyplot as plt

F = h5py.File('output/se_clamp_report.h5', 'r')
report = F['data']
plt.plot(report)
plt.ylabel("current")
plt.xlabel("time")
plt.title("Voltage clamp on AAC 31385 at -70mV")
plt.show()