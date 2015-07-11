from __future__ import division
import cv2
import numpy as n
import numpy.random as rnd
import warnings


class RansacFeature(object):
    '''
    Class for feature detection inside images and videos with
    the RANSAC algorithm
    
    Attributes:
        feature: the feature class
        max_it: Max Number of iterations for the RANSAC loop
        inliers_percent: percentage of inliers over total points.\ 
                         Used for stop criteria in the RANSAC loop
        threshold: Value of threshold used after the image normalization \
                   (can be an integer from 1 to 254)
        dst: the distance of the inliers pixels from the feature (i.e.\
             a pixel is considered an inlier if its distance is < dst)
    '''
    
    def __init__(self,feature,max_it=100,inliers_percent=0.6, threshold = 100, dst = 10):
        self.feature = feature
        self.max_it = max_it 
        self.inliers_percent = inliers_percent 
        self.threshold = threshold
        self.dst = dst
    
    def image_search(self,image):
        ''' This method look for the feature inside a grayscale image
        
        Args:
            image: the image where to detect the circle.
            min_points: the minimum number of points to retrieve the \
                        feature (i.e. 3 points for circle)
            
        Returns:
            feature: The detected feature object
            percent: The percentage of "fitness" (i.e inliers/total_points) of the feature detected
                     in the image
            
        Raises:
            ValueError: If the thresholded image is completely empty (all pixels intensities
                        == 0, a ValueError is raised)
        '''
        
        #=======================================================================
        # TODO: Check if the image is gray
        #=======================================================================

        
        # Normalization
        image = cv2.normalize(image,image, alpha=0,norm_type=cv2.NORM_MINMAX, beta = 255)
        
        # Thresholding
        _ret,image = cv2.threshold(image,self.threshold,255,cv2.THRESH_BINARY)
        
        # Only the non-zero pixels
        pixels = n.where(image>0)
        
        #Thresholded image can be empty
        if not(pixels):
            raise ValueError('Thresholded image is completely empty.\
                            The threshold argument is too high or the image\
                            is totally black')
        
        # Orienting correctly the points in a (n,2) shape
        # needed because of arguments of feature.points_distance()
        pixels = n.transpose(n.vstack([pixels[0],pixels[1]]))
        
        # -- Starting Loop -- #
        
        # Pre-allocating guess points 
        pts = n.zeros((self.feature.min_points,2))
        
        # Starting iterations
        it = 0
        
        # Current percent of inliers over the total points
        percent = 0
        

        while not(percent>self.inliers_percent or it>self.max_it):
            
            # Guess three pixels from the non-zero ones
            pts = pixels[rnd.randint(n.size(pixels[:,0]),size=self.feature.min_points)]
            
          
            
            # Generating Circle from the three given points
            try:
                guess_feature = self.feature(pts)
            except RuntimeError: # If the three points are collinear the circle cannot be computed
                warnings.warn('Probable collinear points for circle')
                continue
            # Compute distance of all non-zero points from the circumference 
            distances = guess_feature.points_distance(pixels)
            
            # Check which points are inliers (i.e. near the circle)
            inliers = n.size(pixels[distances <= self.dst])
            
            #Compute the percentage
            percent_new = inliers/n.size(pixels)
            
            it = it+1
            
            if percent_new > percent:
                # Update if better approximation
                percent = percent_new
                feature = guess_feature
        
                
        #=======================================================================
        # if it >self.max_it:
        #     warnings.warn('''Max Iterations number reached. The current percentage of fitness is {0}'''\
        #                   .format(percent),RuntimeWarning)
        #=======================================================================s
        return (feature,percent)
    
    def video_processing(self,videofile):
        video = cv2.VideoCapture(videofile)
        nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #Pre-allocating dataset for feature array
        fs = n.empty(nframes,dtype=self.feature)
        
        
        for i in xrange(nframes):
            succ, frame = video.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            if succ: # If successfully got the video frame
                
                try:
                    feature,_percent = self.image_search(frame)
                    fs[i] = feature
                except ValueError:
                    fs[i] = None
                
            else:
                raise RuntimeError("Error in retrieving video frames.")
            
        return fs        