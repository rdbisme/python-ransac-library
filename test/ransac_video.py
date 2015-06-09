import tra.ransac as rns
import tra.features as fts

ransac = rns.RansacFeature(fts.Circle)
circles = ransac.video_processing('../video/01_CMP.avi')