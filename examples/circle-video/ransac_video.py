import tra.ransac as rns
import tra.features as fts
import numpy as n


ransac = rns.RansacFeature(fts.Circle,max_it=100,threshold=100,dst=1)
circles = ransac.video_processing('../../video/14_XVID.avi',save_frames=True)

circles = [[c.radius,c.xc,c.yc] for c in circles if c]

n.save('data.npy',circles)