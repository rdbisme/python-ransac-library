from __future__ import division
import numpy as n
import scipy.linalg as linalg

class Feature(object):
    '''
    Abstract class that represents a feature to be used
    with the RansacFeature class in tra.ransac module
    
    Attributes:
        min_points: The minimum points required to compute the feature
                    (e.g. Circle.min_points = 3)
    '''
    
    def __init__(self):
        raise NotImplemented
    
    def points_distance(self,points):
        ''' 
        This function implements a method to compute the distance 
        of points from the feature
        
        Args:
            points: a numpy array of points the distance must be 
                    computed of
        
        Returns: 
            distance: the computed distance of the points from the
                      feature
        '''
        
        raise NotImplemented

class Circle(object):
    ''' 
    This is an helper class for circle-related activities.
    
    Properties:
    - 
    '''
    
    min_points = 3
    
    def __init__(self,points):
        self.radius,self.xc,self.yc = self.__gen(points)
    

    def __gen(self,points):
        '''
        Compute the radius and the center coordinates of a 
        circumference given three points
        
        Args:
            points: a (3,2) numpy array, each row is a 2D Point.
        
        Returns: 
            circle: A (3,) numpy array that contains the circumference radius
            and center coordinates [radius,xc,yc]
            
        Raises: 
            RuntimeError: If the circle computation does not succeed
                a RuntimeError is raised.
            
        
        
    '''            
      
        # Linear system for (D,E,F) in circle 
        # equation: D*x + E*y + F = -(x**2 + y**2)
        
        # Generating A matrix 
        A = n.array([(x,y,1) for x,y in points])
        # Generating rhs
        rhs = n.array([-(x**2+y**2) for x,y in points])
        
        try:
            #Solving linear system
            D,E,F = linalg.lstsq(A,rhs)[0]
        except linalg.LinAlgError:
            raise RuntimeError('Circle calculation not successful. Please\
             check the input data, probable collinear points')
            
        xc = -D/2
        yc = -E/2
        r = n.sqrt(xc**2+yc**2-F)

        return [r,xc,yc]
            
    def points_distance(self,points):
        d = n.abs(\
                  n.sqrt(\
                         n.power(self.xc - points[:,0],2) + n.power(self.yc - points[:,1],2)
                         )\
                  - self.radius
                  )
        return d
