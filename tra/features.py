from __future__ import division
import numpy as n
import scipy.linalg as linalg
import scipy.optimize as opt
import scipy.spatial.distance as dist

class Feature(object):
    '''
    Abstract class that represents a feature to be used
    with :py:class:`tra.ransac.RansacFeature`
    '''
    
    def __init__(self):
        raise NotImplemented
    
    @property
    def min_points(self):
        '''int: Minimum number of points needed to define the feature.'''
        
        raise NotImplemented
    
    def points_distance(self,points):
        ''' 
        This function implements a method to compute the distance 
        of points from the feature.
        
        Args:
            points: a numpy array of points the distance must be 
                    computed of.
        
        Returns: 
            distances (numpy.ndarray): the computed distances of the points from the feature.
        '''
        
        raise NotImplemented

class Circle(Feature):
    ''' 
    Feature class for a Circle :math:`(x-x_c)^2 + (y-y_c)^2 - r = 0`
    '''
    
    min_points = 3
    '''int: Minimum number of points needed to define the circle (3).'''
    
    def __init__(self,points):
        self.radius,self.xc,self.yc = self.__gen(points)
    

    def __gen(self,points):
        '''
        Compute the radius and the center coordinates of a 
        circumference given three points
        
        Args:
            points (numpy.ndarray): a (3,2) numpy array, each row is a 2D Point.
        
        Returns: 
            (tuple): A 3 elements tuple that contains the circumference radius
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

        return (r,xc,yc)
            
    def points_distance(self,points):
        r'''
        Compute the distance of the points from the feature
        
        :math:`d = \left| \sqrt{(x_i - x_c)^2 + (y_i-y_c)^2} - r \right|`
        
        Args:
            points (numpy.ndarray): a (3,2) numpy array, each row is a 2D Point.
            
        Returns:
            d (numpy.ndarray): the computed distances of the points from the feature.
        
        '''
        
        xa = n.array([self.xc,self.yc]).reshape((1,2))
        d = n.abs(dist.cdist(points,xa) - self.radius)
        return d
    
class Exponential (Feature):
    '''
    Feature Class for an exponential curve :math:`y=ax^{k} + b`
    '''
    
    min_points = 3
    
    def __init__(self,points):
        self.a,self.k,self.b = self.__gen(points)
    

    def __gen(self,points):
        '''
        Compute the three parameters that univocally determine the
        exponential curve
        
        Args:
            points(numpy.ndarray): a (3,2) numpy array, each row is a 2D Point.
        
        Returns: 
            exp(numpy.ndarray): A (3,) numpy array that contains the a,n,b parameters
            [a,k,b]
            
        Raises: 
            RuntimeError: If the circle computation does not succeed
                a RuntimeError is raised.
            
        
        
    '''
        def exponential(x,points):
            ''' Non linear system function to use 
            with :py:func:`scypy.optimize.root`
            '''
            aa = x[0]
            nn = x[1]
            bb = x[2]
            
            f = n.zeros((3,))
            f[0] = n.abs(aa)*n.power(points[0,0],nn)+bb - points[0,1]
            f[1] = n.abs(aa)*n.power(points[1,0],nn)+bb - points[1,1]
            f[2] = n.abs(aa)*n.power(points[2,0],nn)+bb - points[2,1]
            
            return f
        

        exp = opt.root(exponential,[1,1,1],points,method='lm')['x']
        return exp

            
    def points_distance(self,points):
        r'''
        Compute the distance of the points from the feature
        
        :math:`d = \sqrt{(x_i - x_c)^2 + (y_i-y_c)^2}`
        
        Args:
            points (numpy.ndarray): a (3,2) numpy array, each row is a 2D Point.
            
        Returns: 
            d (numpy.ndarray): the computed distances of the points from the feature.
        
        '''
        x = points[:,0]
        xa = n.array([x,self.a*n.power(x,self.k)+self.b])
        xa = xa.T
        d = dist.cdist(points,xa)        
        return n.diag(d)
    
    
