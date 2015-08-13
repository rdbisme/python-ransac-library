import tra.features as fts
import numpy as np
import multiprocessing as mp
import time

points = np.random.random(size=(1000,2))
circle_points = np.random.random(size=(3,2))

feature = fts.Circle(circle_points)

t = time.time()
ds_s = feature.points_distance(points,pool=None)
pool = mp.Pool()
t1 = time.time() - t

t = time.time()
ds = feature.points_distance(points,pool=pool,chunks_num=1)
t2 = time.time() - t

print (t1-t2)/t1