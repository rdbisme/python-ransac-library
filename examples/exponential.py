import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from tra import features as ffs
a = 1
b = 1
n = 3
x = np.linspace(2,10,11)
y = a*np.power(x,n)+b
i = [0,3,5]
points = np.array([
                   pps for pps in zip(x[i],y[i])
                   ])
                   
print points

exp = ffs.Exponential(points)
p = [exp.a,exp.k,exp.b]
plt.plot(x,y)
plt.plot(x[i],y[i],'*')
plt.plot(x,p[0]*np.power(x,p[1])+p[2],'o-')
plt.show()