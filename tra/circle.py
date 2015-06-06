#import numpy as n
from scipy import optimize as opt
import numpy as n

class Circle(object):
    ''' 
    This is an helper class for circle-related activities.
    
    Properties:
    - 
    '''
    def __init__(self,points):
        self.radius,self.xc,self.yc = self.__gen(points)
    

    def __gen(self,points):
        '''
        Compute the radius and the center coordinates of a 
        circumference given three points
        
        Args:
            points: a (3,2) numpy array, each row is a 2D Point
        
        Returns: 
            circle: A (3,) numpy array that contains the circumference radius
            and center coordinates.
            
        Raises: 
            RuntimeError: If the circle computation does not succeed
                a RuntimeError is raised.
            
        
        
    '''
        pt1 = points[0]
        pt2 = points[1]
        pt3 = points[2]
            
      
        def __circle(x,pt1,pt2,pt3):
            r  = x[0]
            xc = x[1]
            yc = x[2]
            
            return [
                    (pt1[0]-xc)**2 + (pt1[1]-yc)**2  - r**2,
                    (pt2[0]-xc)**2 + (pt2[1]-yc)**2  - r**2,
                    (pt3[0]-xc)**2 + (pt3[1]-yc)**2  - r**2
                    ]
        
        def __jacobian(x,pt1,pt2,pt3):
            r = x[0]
            xc = x[1]
            yc = x[2]
            
            F = n.array([
                         [-2*r,-2*(pt1[0]-xc), -2*(pt1[1]-yc)],
                         [-2*r,-2*(pt2[0]-xc), -2*(pt2[1]-yc)],
                         [-2*r,-2*(pt3[0]-xc), -2*(pt3[1]-yc)]
                         ])
            return F
        
        #Initial Guess for root
        X0 = [
              n.sum((pt1,pt2,pt3)),
              n.mean((pt1[0],pt2[0],pt3[0])),
              n.mean((pt1[1],pt2[1],pt3[1]))
              ]
    
        circle = opt.root(__circle,X0,args=(pt1,pt2,pt3),\
                          method='lm',jac=__jacobian)
    
        
        if circle['success']:
            return circle['x']
        else:
            raise RuntimeError('Circle calculation not successful. Please\
            please check the input data, probable collinear points')
            
    def points_distance(self,points):
        d = n.abs(\
                  n.sqrt(\
                         n.power(self.xc - points[:,0],2) + n.power(self.yc - points[:,1],2)
                         )\
                  - self.radius
                  )
        return d
