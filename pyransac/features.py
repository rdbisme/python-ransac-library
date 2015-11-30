from __future__ import division
import abc
import numpy as n
import scipy.linalg as linalg
import scipy.optimize as opt
import scipy.spatial.distance as dist

class Feature(object):
    '''
    Abstract class that represents a feature to be used
    with :py:class:`pyransac.ransac.RansacFeature`
    '''
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def __init__(self):
        pass
    
    @abc.abstractproperty
    def min_points(self):
        '''int: Minimum number of points needed to define the feature.'''
        
        pass
    
    @abc.abstractmethod
    def points_distance(self,points):
        ''' 
        This function implements a method to compute the distance 
        of points from the feature.
        
        Args:
            points (numpy.ndarray): a numpy array of points the distance must be 
                    computed of.
        
        Returns: 
            distances (numpy.ndarray): the computed distances of the points from the feature.
        '''
        
        pass
    
    @abc.abstractmethod
    def print_feature(self,num_points):
        '''
        This method returns an array of x,y coordinates for
        points that are in the feature.
        
        Args:
            num_points (numpy.ndarray): the number of points to be returned
            
        Returns:
            coords (numpy.ndarray): a num_points x 2 numpy array that contains 
            the points coordinates  
        '''

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
        # equations: D*xi + E*yi + F = -(xi**2 + yi**2)
        # where xi, yi are the coordinate of the i-th point.
        
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
    
    
    def print_feature(self, num_points):
        '''
        This method returns an array of x,y coordinates for
        points that are in the feature.
        
        Args:
            num_points (numpy.ndarray): the number of points to be returned
            
        Returns:
            coords (numpy.ndarray): a num_points x 2 numpy array that contains 
            the points coordinates  
        '''
        
        theta = n.linspace(0,2*n.pi,num_points)
        x = self.xc + self.radius*n.cos(theta)
        y = self.yc + self.radius*n.sin(theta)
        
        return n.vstack((x,y))
    
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
    
    def print_feature(self, num_points, a,b):
        '''
        This method returns an array of x,y coordinates for
        points that are in the feature in the interval [a,b].
        
        Args:
            num_points (numpy.ndarray): the number of points to be returned
            a (float): left end of the interval
            b (float): right end of the interval
            
        Returns:
            coords (numpy.ndarray): a num_points x 2 numpy array that contains 
            the points coordinates  
        '''
        
        
        x = n.linspace(a,b,num_points)
        y = self.a*x**self.k + self.b
        
        return n.vstack((x,y))
    
