from __future__ import division
import cv2
import numpy as n
from numpy import random as rnd
from matplotlib import pyplot as plt 
from tra import circle as c

video = cv2.VideoCapture('../video/01_CMP.avi')
video.set(cv2.CAP_PROP_POS_FRAMES,200)
succ, frame = video.read()

frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
frame = cv2.normalize(frame,frame, alpha=0,norm_type=cv2.NORM_MINMAX, beta = 255)
ret,frame = cv2.threshold(frame,40,255,cv2.THRESH_BINARY)

points = n.where(frame>0) #Thresholded pixels

#Orienting correctly the points in a (n,2) shape
#needed because of arguments of circle.points_distance()
points = n.transpose(n.vstack([points[0],points[1]]))


inliers_old = 0
for it in xrange(10000):
    #Get 3 Random points from the frame
    h,w=n.shape(frame)
    pts = n.zeros((3,2))
    pts[:,0] = rnd.randint(w,size=3)
    pts[:,1] = rnd.randint(h,size=3)
     
    
    #Generating Circle
    try:
        circle = c.Circle(pts)
    except RuntimeError:
        break
    
    distances = circle.points_distance(points)
     
    # Counting all the points near (with a threshold) the circle == inliers
    tresh_dist = 3
    inliers = n.size(points[distances <= tresh_dist])
    if inliers > inliers_old:
        cc = circle
        cs = [circle.radius,circle.xc,circle.yc]
        inliers_old = inliers
        pps = pts
         
 
 

theta = n.linspace(-n.pi,n.pi,100)
plt.plot(pps[:,0],pps[:,1],'o')
plt.plot(points[:,0],points[:,1],'wo')
plt.axis('equal')
plt.plot(cs[1] + cs[0]*n.cos(theta),cs[2]+cs[0]*n.sin(theta),'r-',linewidth=2)
#plt.imshow(frame, cmap=plt.get_cmap('gray'),extent=[0,640,0,480],aspect=2)

 
plt.show()
video.release()