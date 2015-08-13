from __future__ import division
import copy_reg
import multiprocessing as mp
import numpy as n
import scipy.linalg as linalg
import scipy.optimize as opt
import scipy.spatial.distance as dist
import types

def _pickle_method(method):
    # Author: Steven Bethard
    # http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    cls_name = ''
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
    if cls_name:
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    # Author: Steven Bethard
    # http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

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
            
    def __points_distance(self,points):
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
    
    def points_distance(self,points,pool=None):
        if pool:
            
            #Manual Chunking. Not efficient!
            ## TODO ##
            #Optimize Chunking
            chunks_num = 10*mp.cpu_count()
            chunks = n.array_split(points,chunks_num)
            return n.vstack(pool.map(self.__points_distance,chunks))
            
        else:
            return self.__points_distance(points)

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

