from __future__ import division
import cv2
import numpy as n

video = cv2.VideoCapture('../video/01_CMP.avi')
video.set(cv2.CAP_PROP_POS_FRAMES,500)
succ, frame = video.read()

frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
frame = cv2.normalize(frame,frame, alpha=0,norm_type=cv2.NORM_MINMAX, beta = 255)
ret,frame = cv2.threshold(frame,50,255,cv2.THRESH_BINARY_INV)

cv2.imshow('frame',frame)
cv2.waitKey(0)

video.release()
cv2.destroyAllWindows()