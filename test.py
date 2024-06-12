from IPython import embed
from RVBUST import Vis
import numpy as np
import open3d as o3d
import os
from VisionTools import Utils
import matplotlib.pyplot as plt
import cv2

v = Vis.View()

# def FitPlane(pc, threshold=0.001, iteration=1000):
#     para, inlier = Utils.NumpyToPCD(pc).segment_plane(
#         distance_threshold=threshold, ransac_n=3, num_iterations=iteration)
#     return para, inlier

def ProjectPointsToPlane(points,plane,planeCenter):
    print('ProjectPointsToPlane')
    points[np.where(np.isnan(points[:,0]))] = [0,0,0]
    rz = plane[:3]
    if plane[2]<0:
        rz = -rz
    ry = np.cross(rz, [1, 0, 0])
    rx = np.cross(ry, rz)
    rx = rx/np.linalg.norm(rx)
    ry = ry/np.linalg.norm(ry)
    rz = rz/np.linalg.norm(rz)
    R = np.c_[rx, ry, rz]
    points = (R.T@points.T).T      #将点云垂直投影于拟合的平面
    plane_center = R.T@planeCenter
    return points,plane_center,R


def GetDepthImage(points,ImageHeight:int,ImageWidth:int):
    if points.size == ImageHeight*ImageWidth*3:
        xDepth = np.asarray(points[:,0]).reshape(ImageHeight,ImageWidth)
        yDepth = np.asarray(points[:,1]).reshape(ImageHeight,ImageWidth)
        zDepth = np.asarray(points[:,2]).reshape(ImageHeight,ImageWidth)
        return True,xDepth,yDepth,zDepth
    else:
        return False,None,None,None
        
def RemoveNaNs(pm):
    pm[np.where(np.isnan(pm[:,0]))] = [0,0,0]
    return pm

def find_plane(pts, threshold=0.0003):
    pts = pts[np.where(pts[:,2]!=0)]
    plane_model,inliers = Utils.FitPlane(pts, threshold=threshold)
    plane_points = pts[inliers]
    # 提取平面外的点
    non_plane_points = pts[~np.isin(np.arange(len(pts)), inliers)]
    plane_center = np.mean(plane_points,axis=0)
    return plane_model, plane_center ,plane_points,non_plane_points

def read_single_point_cloud(path = 'PointCloud.ply'):
    point_cloud = o3d.io.read_point_cloud('PointCloud.ply')

    pts = Utils.PCDToNumpy(point_cloud)
    pts = RemoveNaNs(pts)
    return pts

point_cloud = read_single_point_cloud()
point_cloud[np.asarray(point_cloud[:,2] <75)] = [0,0,0]
point_cloud[np.asarray(point_cloud[:,2] > 100)] = [0,0,0]
point_cloud[np.asarray(point_cloud[:,0] < 30)] = [0,0,0]
point_cloud[np.asarray(point_cloud[:,0] > 730)] = [0,0,0]
point_cloud[np.asarray(point_cloud[:,1] < 100)] = [0,0,0]
ptsplane_model, plane_center ,plane_points,non_plane_points = find_plane(pts=point_cloud,threshold= 3)
points,plane_center,R = ProjectPointsToPlane(point_cloud,ptsplane_model,plane_center)
points[np.asarray(points[:,2] > plane_center[2] + 3)] = [0,0,0]
points[np.asarray(points[:,2] < plane_center[2] - 3)] = [0,0,0]
embed()
ptsplane_model, plane_center ,plane_points,non_plane_points = find_plane(pts=plane_points,threshold= 2)

ptsplane_model, plane_center ,plane_points,non_plane_points = find_plane(pts=non_plane_points,threshold= 3)
