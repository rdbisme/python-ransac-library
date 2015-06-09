import tra.ransac as rns
import tra.features as fts
import numpy as n
import time as t

tt = t.time()
ransac = rns.RansacFeature(fts.Circle,max_it=100,threshold=230,dst=1)
circles = ransac.video_processing('../video/02_CMP.avi')
circles = [[c.radius,c.xc,c.yc] for c in circles if c]

n.save('data.npy',circles)

print t.time() -tt