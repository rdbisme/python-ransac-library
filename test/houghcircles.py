from __future__ import division 
import cv2
import numpy as n
from matplotlib import pylab as plt
from scipy import signal as sig
int

video = cv2.VideoCapture('../video/01_CMP.avi')
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
fps = video.get(cv2.CAP_PROP_FPS)
size = ((int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

out = cv2.VideoWriter('../video/01_CMP_out.mov',
                      fourcc,
                      fps,size)

nframes = video.get(cv2.CAP_PROP_FRAME_COUNT)


radguess=[100]
mean = int(n.round(n.mean(radguess)))
std = 10
while(True):
    succ,frame = video.read()
    if succ:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #frame2 = cv2.medianBlur(frame,31)
        frame = cv2.GaussianBlur(frame,(11,11),0)
        frame2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                       cv2.THRESH_BINARY_INV, \
                                       3,1)
        
        cs = cv2.HoughCircles(frame2,\
                                   method=cv2.HOUGH_GRADIENT,\
                                   dp=4,\
                                   minDist=20,\
                                   param1=30, 
                                   param2=15,\
                                   minRadius=mean-std,
                                   maxRadius=mean+std)
        
        if not(cs == None):
            cv2.circle(frame2,(cs[0,0,0],cs[0,0,1]),cs[0,0,2],(255,255,255),2)
            cv2.imshow('frame',frame2)
            if not(cv2.waitKey(1) == ord('q')):
                radguess.append(cs[0,0,2])
                mean = int(n.round(n.mean(radguess)))
                std = int(n.round(0.45*mean))
                #out.write(frame)
            else:
                break;
    else:
        break

        

'''
video.set(cv.CV_CAP_PROP_POS_FRAMES,n.round(nframes/2)-30)
succ,frame = video.read()


frame = cv2.medianBlur(frame,5)
frame = cv2.cvtColor(frame,cv.CV_RGB2GRAY)
cs = cv2.HoughCircles(frame,
                           method=cv2.cv.CV_HOUGH_GRADIENT,
                           dp=2,
                           minDist=20,
                           param1=50, 
                           param2=30,
                           minRadius=100,
                           maxRadius=0)

#print n.shape(circles[0])
cs = cs[0,:]
#imax = cs[:,2].argmax()
imax = 0
cv2.circle(frame,(cs[imax,0],cs[imax,1]),cs[imax,2],(255,255,255),2)
    
cv2.imshow('frame',frame)
cv2.waitKey(5000)'''
# When everything done, release the capture
video.release()
out.release()
cv2.destroyAllWindows()
b,a = sig.butter(8, 0.03, 'low', output='ba')
radguess = sig.filtfilt(b, a, radguess)
plt.plot(radguess)
plt.show()