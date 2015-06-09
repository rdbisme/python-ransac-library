from __future__ import division
import numpy as n
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.optimize as opt

pixel_to_mm = 8/350
frame_to_time = 1/250

circles = n.load('data.npy')
circles = circles[circles[:,0]>70]
circles = circles[circles[:,0]<500]

b,a = sig.butter(8,0.01, 'low', output='ba')
filtered = sig.filtfilt(b, a, circles[:,0])

def opt_func(x,a,b,n):
    return a*x**n + b

x = range(n.size(filtered))

popt,_P = opt.curve_fit(opt_func, x, circles[:,0])
y = opt_func(x,*popt)

plt.plot(x,circles[:,0],'-.',mfc='none')
#plt.plot(x,filtered,'g-',linewidth=5)
plt.plot(x,y,'r-')
plt.xlabel('Frames No.')
plt.ylabel('Radius [pixels]')
plt.show()