from IPython import embed
from RVBUST import Vis
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from VisionTools import Utils
from scipy.spatial import KDTree,cKDTree
import math
from WeldPolishing0811 import WeldSeamPolishing

# 填写手眼标定
HE = np.array([[-0.458624 ,0.888597 ,-0.00769609, -264.467],
[0.886836, 0.458231, 0.0595551, -40.111],
[0.0564471 ,0.0204882 ,-0.998195, 605.464],
[0 ,0 ,0 ,1]])

# 显示窗口
v = Vis.View("test")

def distance_func(x, y):
    return np.abs(x[1] - y[1])  # 计算y值的差的绝对值作为距离

# 创建对象，传入手眼标定信息
welding_polishing = WeldSeamPolishing(hand_eye_cali=HE , view = v)

# 读取底面点云，传出后切割，再传入，至此找到底面点云法向，点云，中心点
bottom_point_clouds_old  = welding_polishing.read_bottom_point_clouds(bottom_file_path = 'test13')
bottom_point_clouds  = welding_polishing.read_bottom_point_clouds(bottom_file_path = 'test25')
bottom_point_clouds[np.asarray(bottom_point_clouds[:,1]<-2.18)] = [0,0,0]
bottom_point_clouds[np.asarray(bottom_point_clouds[:,1]>-2.08)] = [0,0,0]
bottom_point_clouds[np.asarray(bottom_point_clouds[:,2]>-0.22)] = [0,0,0]
bottom_point_clouds[np.asarray(bottom_point_clouds[:,2]<-0.3)] = [0,0,0]
embed()
# 底面，正面，以及交线
ret = welding_polishing.processing_bottom_points(bottom_point_clouds)

#读取上面点云
#top_point_clouds  = welding_polishing.read_top_point_clouds(top_file_path = 'test15')
top_point_clouds  = welding_polishing.read_top_point_clouds(top_file_path = 'test25')
top_point_clouds[np.asarray(top_point_clouds[:,1]<-2.17)] = [0,0,0]
top_point_clouds[np.asarray(top_point_clouds[:,1]>-2.08)] = [0,0,0]
top_point_clouds[np.asarray(top_point_clouds[:,2]>-0.03)] = [0,0,0]
top_point_clouds[np.asarray(top_point_clouds[:,2]<-0.08)] = [0,0,0]
ret = welding_polishing.processing_top_points(top_point_clouds)
#ret = welding_polishing.find_top_plane(top_point_clouds)

# 读取左面点云，左面点云包含，前面，左侧面
#left_point_clouds = welding_polishing.read_left_point_clouds(left_file_path = 'test8')
left_point_clouds = welding_polishing.read_left_point_clouds(left_file_path = 'test28')
front_point_clouds = left_point_clouds.copy()
front_point_clouds[np.asarray(front_point_clouds[:,1]<-2.1)] = [0,0,0]
front_point_clouds[np.asarray(front_point_clouds[:,1]>-0.1)] = [0,0,0]
ret = welding_polishing.find_front_plane_l(front_point_clouds)

# 找前面和底面，上面交线
# ret = welding_polishing.find_left_front_t_line()
# ret = welding_polishing.find_left_front_b_line()

# 左侧面点云
lside_point_clouds = welding_polishing.left_point_clouds.copy()
lside_point_clouds[np.asarray(lside_point_clouds[:,1]<-2.195)] = [0,0,0]
lside_point_clouds[np.asarray(lside_point_clouds[:,0]<0.35)] = [0,0,0]
lside_point_clouds[np.asarray(lside_point_clouds[:,0]>0.42)] = [0,0,0]
ret = welding_polishing.find_left_plane(lside_point_clouds)
# ret = welding_polishing.find_left_t_line()
# ret = welding_polishing.find_left_b_line()

# #剩余点云
# ret = welding_polishing.find_left_surface()

# 这里是全部的左面点云
left_point_clouds[np.asarray(left_point_clouds[:,1]<-2.195)] = [0,0,0]
left_point_clouds[np.asarray(left_point_clouds[:,1]>-1.95)] = [0,0,0]
left_point_clouds[np.asarray(left_point_clouds[:,0]<0.15)] = [0,0,0]# 0.2-.3
left_point_clouds[np.asarray(left_point_clouds[:,0]>0.42)] = [0,0,0]
left_point_clouds[np.asarray(left_point_clouds[:,2]>-0.03)] = [0,0,0]
left_point_clouds[np.asarray(left_point_clouds[:,2]<-0.3)] = [0,0,0]
ret = welding_polishing.processing_left_point_clouds()

ret = welding_polishing.find_left_weld_seam()
# 左侧轨迹



# 读取右面点云
#right_point_clouds = welding_polishing.read_right_point_clouds(right_file_path = 'test19')
right_point_clouds = welding_polishing.read_right_point_clouds(right_file_path = 'test29')
front_point_clouds = right_point_clouds.copy()
front_point_clouds[np.asarray(front_point_clouds[:,1]<-2.07)] = [0,0,0]
front_point_clouds[np.asarray(front_point_clouds[:,1]>-0.1)] = [0,0,0]
front_point_clouds[np.asarray(front_point_clouds[:,2]>-0.03)] = [0,0,0]
front_point_clouds[np.asarray(front_point_clouds[:,2]<-0.3)] = [0,0,0]
ret = welding_polishing.find_front_plane_r(front_point_clouds)

# 右侧面点云
rside_point_clouds = right_point_clouds.copy()
rside_point_clouds[np.asarray(rside_point_clouds[:,1]<-2.195)] = [0,0,0]
rside_point_clouds[np.asarray(rside_point_clouds[:,0]<-0.02)] = [0,0,0]
rside_point_clouds[np.asarray(rside_point_clouds[:,0]>+0.05)] = [0,0,0]
ret = welding_polishing.find_right_plane(rside_point_clouds)

# 右侧所有点云
right_point_clouds[np.asarray(right_point_clouds[:,1]<-2.195)] = [0,0,0]
right_point_clouds[np.asarray(right_point_clouds[:,1]>-1.95)] = [0,0,0]
right_point_clouds[np.asarray(right_point_clouds[:,0]<-0.02)] = [0,0,0]# 0.2-.3
right_point_clouds[np.asarray(right_point_clouds[:,0]>0.17)] = [0,0,0]
right_point_clouds[np.asarray(right_point_clouds[:,2]>-0.03)] = [0,0,0]
right_point_clouds[np.asarray(right_point_clouds[:,2]<-0.3)] = [0,0,0]
ret = welding_polishing.processing_right_point_clouds()
v.Point(left_point_clouds,0.5,(1,0,0))
v.Point(right_point_clouds,0.5,(1,0,0))
embed()
#找到交线
welding_polishing.find_right_weld_seam(tolerance = 0.003)

x_coords = np.array(welding_polishing.bottom_intersection_points)[:, 0]

# # 对x坐标进行排序，得到排序后的索引
sorted_indices = np.argsort(x_coords)

# # 根据排序后的索引重新排列点云
welding_polishing.bottom_intersection_points = np.array(welding_polishing.bottom_intersection_points)[sorted_indices]

v.Point(welding_polishing.bottom_intersection_points,8.5,(0,1,0))
v.Point(welding_polishing.right_point_clouds,0.5,(1,0,0))
v.Point(welding_polishing.matching_points,5.5,(1,0,0))

# v.Point(welding_polishing.top_intersection_points,5.5,(0,1,0))

new_array = np.array([[0, 0, 0, 1]])
result = np.concatenate((welding_polishing.bottom_intersection_points,
                         new_array.repeat(welding_polishing.bottom_intersection_points.shape[0], axis=0)), axis=1)

for p in result:
   v.Axes(p[:3],p[3:],0.002,3) 

v.Home()

# 指定要保存的文件路径和文件名
file_path = "result.txt"

# 将结果写入文本文件
np.savetxt(file_path, result, delimiter=",", fmt="%.6f")

v.Point(welding_polishing.max_point_on_surface_top_l,2.5,(0,0,1))
v.Point(welding_polishing.surface_plane_top_inter_points_l_restored,1.5,(0,0,1))
v.Point(welding_polishing.max_point_on_front_bottom_l,10.5,(0,0,1))
v.Point(welding_polishing.max_point_on_front_top_l,10.5,(0,0,1))
v.Point(welding_polishing.max_point_on_left_bottom,10.5,(0,0,1))
v.Point(welding_polishing.max_point_on_left_top,10.5,(0,0,1))
v.Point(welding_polishing.max_point_on_surface_bottom_l,10.5,(0,0,1))
v.Point(welding_polishing.max_point_on_surface_top_l,10.5,(0,0,1))

v.Point(welding_polishing.min_point_on_surface_top_l,2.5,(0,0,1))

v.Point(welding_polishing.min_point_on_front_bottom_l,10.5,(0,0,1))
v.Point(welding_polishing.min_point_on_front_top_l,10.5,(0,0,1))
v.Point(welding_polishing.min_point_on_left_bottom,10.5,(0,0,1))
v.Point(welding_polishing.min_point_on_left_top,10.5,(0,0,1))
v.Point(welding_polishing.min_point_on_surface_bottom_l,10.5,(0,0,1))
v.Point(welding_polishing.min_point_on_surface_top_l,10.5,(0,0,1))


embed()



# lfront_center = welding_polishing.front_center
# # 在点云中查找与给定点 z 值相近的点
# for p in welding_polishing.front_plane_points:
#     if p[0] > lfront_center[0] and abs(p[2]-lfront_center[2]) <= 0.001:
#         lfront_center = p

#       # 计算两个平面法向量的叉积
# line_direction = np.cross(welding_polishing.front_plane_model, welding_polishing.bottom_plane_model)

# # 求解线性方程组以找到交线上的一点
# A = np.vstack((welding_polishing.front_plane_model, welding_polishing.bottom_plane_model))
# b = np.array([-welding_polishing.front_plane_d, -self.left_plane_d])
# point_on_line, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

path1 = np.concatenate((welding_polishing.left_bottom_line_path, welding_polishing.left_bottom_curve_path), axis=0)
path2 = np.concatenate((path1, welding_polishing.front_bottom_line_path), axis=0)
path3 = np.concatenate((path2, welding_polishing.right_bottom_curve_path), axis=0)
path4 = np.concatenate((path3, welding_polishing.right_bottom_line_path), axis=0)
x_ = path4[:,0]
sorted_indices = np.argsort(x_)[::-1]
new2 = np.array(path4)[sorted_indices]
# 指定要保存的文件路径和文件名
file_path = "bottom_path.txt"

# 将结果写入文本文件
np.savetxt(file_path, new2, delimiter=",", fmt="%.6f")

path1_t = np.concatenate((welding_polishing.left_top_line_path, welding_polishing.left_top_curve_path), axis=0)
path2_t = np.concatenate((path1_t, welding_polishing.front_top_line_path), axis=0)
path3_t = np.concatenate((path2_t, welding_polishing.right_top_curve_path), axis=0)
path4_t = np.concatenate((path3_t, welding_polishing.right_top_line_path), axis=0)
x_t = path4_t[:,0]
sorted_indices = np.argsort(x_t)[::-1]
new2_t = np.array(path4_t)[sorted_indices]
# 指定要保存的文件路径和文件名
file_path = "top_path.txt"

# 将结果写入文本文件
np.savetxt(file_path, new2_t, delimiter=",", fmt="%.6f")