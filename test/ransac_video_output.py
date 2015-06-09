from __future__ import division
import numpy as n
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.optimize as opt

pixel_to_mm = 8/350
frame_to_time = 1/250
N = 100

circles = n.load('data.npy')

x = n.where((circles[:,0]>70) & (circles[:,0]<500))[0]
circles = circles[x]

b,a = sig.butter(8,0.05, 'low', output='ba')
filtered = sig.filtfilt(b, a, circles[:,0])


def opt_func(t,a,b,n):
    return a*t**n + b
  


# Scaling

x = n.multiply(x,frame_to_time)
filtered= n.multiply(filtered,pixel_to_mm)
circles = n.multiply(circles,pixel_to_mm)

popt,_P = opt.curve_fit(opt_func, x, circles[:,0])
y = opt_func(x,*popt)
eq = "$r={0:+1.2f}t^{{{2:1.2f}}}{1:+1.2f}$".format(*popt)

plt.plot(x,circles[:,0],'ko',mfc='None',alpha=0.4,label="Computed")
plt.plot(x,filtered,'k--',linewidth=3,label="Filtered")
plt.plot(x,y,'k-',linewidth=2,label="Model fitted")
ax = plt.gca()
ax.text(0.5, 6, eq, fontsize=15)
plt.xlabel('Time [s]')
plt.ylabel('Radius [mm]')
plt.title('02_XVID.avi')
plt.legend()
plt.grid()
plt.show()