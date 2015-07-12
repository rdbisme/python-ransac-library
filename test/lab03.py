import cv2
import tra.ransac as rns
import tra.features as fts
import numpy as n
import glob
import time

t = time.time()
videos = glob.glob('../video/*.avi')

ro = {
         '14_XVID.avi': [
                         5E2,
                         5,
                         1E2
                         ],
         'H10Al01g_250_10_01_G2.avi': [
                                       1E3,
                                       3,
                                       2E2
                                       ],
         'H10Al01i_250_10_01_G2.avi': [
                                       1E3,
                                       1.5,
                                       2E2
                                       ],
         'H10OCAlex_250_10_01_G2.avi': [
                                        1E3,
                                        5,
                                        2E2
                                        ]
      }
                     


data = []
for videopath in videos:
    video = cv2.VideoCapture(videopath)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    name = videopath.split('/')[2]
    ransac = rns.RansacFeature(fts.Circle,max_it=ro[name][0],dst=ro[name][1],threshold=ro[name][2])
    circles = ransac.video_processing(videopath)
    circles = [[c.radius,c.xc,c.yc] for c in circles if c]
    data.append([name,fps,circles])

print time.time()-t
#n.save('data.npy',data)