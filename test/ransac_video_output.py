import numpy as n
import scipy.signal as sig
import matplotlib.pyplot as plt

circles = n.load('data.npy')
circles = circles[circles[:,0]>70]

b,a = sig.butter(8,0.01, 'low', output='ba')
filtered_circles = sig.filtfilt(b, a, circles[:,0])

plt.plot(circles[:,0],'-.',mfc='none')
plt.plot(filtered_circles,'r-',linewidth=5)
plt.xlabel('Frames No.')
plt.ylabel('Radius [pixels]')
plt.show()