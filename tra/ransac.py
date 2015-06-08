from __future__ import division
import cv2
from circle import Circle as c
from matplotlib import pyplot as plt
import numpy as n
from numpy import random as rnd
import warnings

class RansacCircle(object):
    '''
    This class provides useful methods for circle detecting inside
    images or videos
    
    '''
    
    def __init__(self,max_it=100,inliers_percent=0.6, threshold = 40, dst = 10):
        self.max_it = max_it # Max number of iterations for RANSAC loop
        self.inliers_percent = inliers_percent # Percentage of inliers over total points
        self.threshold = threshold # Value of threshold after normalization of the image (1 - 254)
        self.dst = dst #The distance of the points from the random circle
        
    def image_search(self,image):
        ''' This method look for circle inside a grayscale image
        
        Args:
            image: the image where to detect the circle.
            
        Returns:
            circle: A (3,) numpy array that contains the circle radius
            and center coordinates.
        '''
        
        #=======================================================================
        # TODO check if the image is really a grayscale
        #=======================================================================
        #Width and Height of the video
        h,w = n.shape(image)
        
        # Normalization
        image = cv2.normalize(image,image, alpha=0,norm_type=cv2.NORM_MINMAX, beta = 255)
        
        # Thresholding
        ret,image = cv2.threshold(image,self.threshold,255,cv2.THRESH_BINARY)
        
        # Only the non-zero pixels
        pixels = n.where(image>0)
        
        # Orienting correctly the points in a (n,2) shape
        # needed because of arguments of Circle.points_distance()
        pixels = n.transpose(n.vstack([pixels[0],pixels[1]]))
        
        # -- Starting Loop -- #
        # Pre-allocating guess points 
        pts = n.zeros((3,2))
        
        # Starting iterations
        it = 0
        
        # Current percent of inliers over the total points
        percent = 0
        

        while not(percent>self.inliers_percent or it>self.max_it):
            
            # Guess three pixels from the non-zero ones
            pts = pixels[rnd.randint(n.size(pixels[:,0]),size=3)]
            
          
            
            # Generating Circle from the three given points
            try:
                guess_circle = c(pts)
            except RuntimeError: # If the three points are collinear the circle cannot be computed
                warnings.warn('Probable collinear points for circle')
                continue
            # Compute distance of all non-zero points from the circumference 
            distances = guess_circle.points_distance(pixels)
            
            # Check which points are inliers (i.e. near the circle)
            inliers = n.size(pixels[distances <= self.dst])
            
            #Compute the percentage
            percent_new = inliers/n.size(pixels)
            
            it = it+1
            
            if percent_new > percent:
                # Update if better approximation
                percent = percent_new
                circle = guess_circle
        
                
        if it >self.max_it:
            warnings.warn('''Max Iterations number reached. 
                            The current percentage of detection is {0}'''\
                          .format(percent),RuntimeWarning)
        return (circle,percent)
    
            
    def video_processing(self,videofile,plots = False):
        video = cv2.VideoCapture(videofile)
        nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #Pre-allocating dataset for circles
        cs = n.zeros((nframes,3))
        
        if plots:
            # Pre allocating theta angle for plotting circles
            theta = n.linspace(-n.pi,n.pi,100)
        
        for i in xrange(nframes):
            succ, frame = video.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if succ: #If successfully got the video frame
                
                cs[i] = self.image_search(frame)
                
                if plots:
                    plt.imshow(frame, cmap='gray')
                    plt.plot(cs[i].yc + cs[i].radius*n.cos(theta), cs[i].xc + cs[i].radius*n.sin(theta),'r-',linewidth=2)
            else:
                raise RuntimeError("Error in retrieving video frames.")
        return cs
            
            
            
            
            