from __future__ import print_function
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pyransac import features as ffs

# Generating an exponential function
start = 2
end = 10
num_points = 11

a = 1
b = 1
n = 3
x = np.linspace(a,10,11)
y = a*np.power(x,n)+b

# Select three points from the mathematical
# function
i = [0,3,5]
points = np.array([
                   pps for pps in zip(x[i],y[i])
                   ])
                   

# Generate Exponential feature from the 
# three points
exp = ffs.Exponential(points)

# Compare parameters
print("""
      Manual set parameters for exponential y=ax*n + b:\n
      a = {0}
      b = {1}
      k = {2}
      
      Detected feature parameters:
      
      a = {3}
      b = {4}
      k = {5}
      """.format(a,b,n,exp.a,exp.b,exp.k)
      )

# Print feature
xe,ye = exp.print_feature(num_points,start,end)
plt.plot(x,y)
plt.plot(x[i],y[i],'*')
plt.plot(xe,ye,'o-')
plt.show()