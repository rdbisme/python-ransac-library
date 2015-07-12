import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.signal as sig
import cPickle
from tra import ransac,features

def opt_func(t,a,b,n):
    return a*t**n + b

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

pixel_to_mm = 1

data = cPickle.load(open('data.p','rb'))

b,a = sig.butter(6,0.05, 'low', output='ba')

tigns = [0.15,6,8.25,10.5]
tstop = [2,50,40,50]

popt = [1,1,1]
for i,d in enumerate(data):
    print i
    plt.figure()
    values = np.array(d[2])
    points = values[:,0]
    frames = xrange(len(points))
    frames = np.where((points>80) & (points<200))
    points = points[frames]
    frames = np.reshape(frames,(len(points),))
    
    #Calibration
    fps = d[1]
    t = frames/fps
    
    # Setting Time of ignition
    l = np.where((t>tigns[i]) & (t<tstop[i]))
    t = t[l]
    points = points[l]
    
    # Filtering
    filtered = sig.filtfilt(b, a, points)
    #filtered = smooth(points,100)

    
    #Optimization Fitting
    #popt,_P = opt.curve_fit(opt_func, t, filtered,p0=popt)
    #y = opt_func(t,*popt)
    rnsc = ransac.RansacFeature(features.Exponential,dst=0.8,max_it=500)
    pixels = np.array([
                      t,points
                      ]).T
    exp_f,_ = rnsc.compute_feature(pixels)
    y = exp_f.a*np.power(t,exp_f.k)+ exp_f.b
    eq = "$r={0:+1.2f}t^{{{2:1.2f}}}{1:+1.2f}$".format(*popt)
    
    plt.plot(t,points,'o',mfc='0.4',alpha=0.3)
    plt.plot(t,filtered,'k--')
    plt.plot(t,y,'k-')
    plt.ylim([70,200])
    
plt.show()
