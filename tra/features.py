from __future__ import division
import copy_reg
import numpy as n
import scipy.linalg as linalg
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
    with the RansacFeature class in tra.ransac module
    
    Attributes:
        min_points: The minimum points required to compute the feature
                    (e.g. Circle.min_points = 3)
    '''
    
    def __init__(self):
        raise NotImplemented
    
    @property
    def min_points(self):
        '''Min number of points to define the feature'''
    
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

class Circle(Feature):
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
            
    def __points_distance(self,points):
        xa = n.array([self.xc,self.yc]).reshape((1,2))
        d = n.abs(dist.cdist(points,xa) - self.radius)
        return d
    
    def points_distance(self,points):
        return pool.map(self.__points_distance,points)












