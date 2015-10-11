from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as n
from tra.ransac import RansacFeature
from tra.features import Circle

image = cv2.imread('circle.png')
#print image

ransac_process = RansacFeature(Circle,max_it = 100, inliers_percent=0.7,dst=1,threshold=100)
dc,percent = ransac_process.image_search(image)

theta = n.linspace(-n.pi,n.pi,100)
plt.imshow(image, cmap='gray')
plt.plot(dc.yc + dc.radius*n.cos(theta), dc.xc + dc.radius*n.sin(theta),'r-',linewidth=2)
plt.axis('off')
plt.show()