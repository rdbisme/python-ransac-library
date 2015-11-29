from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as n
from pyransac.ransac import RansacFeature
from pyransac.features import Circle

video = cv2.VideoCapture('media/HRE-video.avi')
print video.get(cv2.CAP_PROP_FPS)
video.set(cv2.CAP_PROP_POS_FRAMES,250)
succ, frame = video.read()
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

ransac_process = RansacFeature(Circle,max_it = 1E3, inliers_percent=0.7,dst=3,threshold=100)
dc,percent = ransac_process.image_search(frame)

theta = n.linspace(-n.pi,n.pi,100)
plt.imshow(frame, cmap='gray')
#plt.plot(dc.yc + dc.radius*n.cos(theta), dc.xc + dc.radius*n.sin(theta),'r-',linewidth=2)
plt.ylim([0,512])
plt.xlim([0,512])
plt.axis('off')
plt.show()
video.release()