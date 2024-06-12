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
import transformations
import matplotlib.pyplot as plt


class WeldSeamPolishing:
    def __init__(self,single_file_path = None,multi_files_path = None,hand_eye_cali =None,view = None):
        self.v = view
        self.hand_eye_cali = hand_eye_cali

    def find_bottom_plane(self,pts):
        # 找到底面中心和法向
        #self.bottom_point_clouds = pts.copy()
        bottom_plane_model , self.bottom_center,self.bottom_plane_points, self.front_point_clouds_b=  self.find_plane(self.bottom_point_clouds)
        self.bottom_plane_d = bottom_plane_model[3]
        self.bottom_plane_model = bottom_plane_model[:3]

    def find_front_plane_b(self):
        front_plane_model_b , self.front_center_b,self.front_plane_points_b, _=  self.find_plane(self.front_point_clouds_b)
        self.front_plane_model_d_b = front_plane_model_b[3]
        self.front_plane_model_b = front_plane_model_b[:3]
    
    def find_b_f_line(self):
        self.b_f_line_direction = np.cross(self.front_plane_model_b, self.bottom_plane_model)
    
    def cali_b_f_angle(self):
        angle = np.arccos(np.dot(self.front_plane_model_b, self.bottom_plane_model) / (np.linalg.norm(self.front_plane_model_b) * np.linalg.norm(self.bottom_plane_model)))
        self.bf_angle_degrees = np.degrees(angle)
        if 0 <= self.bf_angle_degrees <=90:
            self.bf_angle_degrees = 180 - self.bf_angle_degrees

    def processing_bottom_points(self,pts):
        self.bottom_point_clouds = pts.copy()
        self.find_bottom_plane(pts)
        self.find_front_plane_b()
        self.find_b_f_line()
        self.cali_b_f_angle()

    def processing_top_points(self,pts):
        self.top_point_clouds = pts.copy()
        self.find_top_plane(pts)
        self.find_t_f_line()
        self.cali_t_f_angle()

    def find_top_plane(self,pts):
        # 找到顶面中心和法向
        top_plane_model , self.top_center,self.top_plane_points ,_=  self.find_plane(self.top_point_clouds)
        self.top_plane_d = top_plane_model[3]
        self.top_plane_model = top_plane_model[:3]

    def find_t_f_line(self):
        self.t_f_line_direction = np.cross(self.front_plane_model_b, self.top_plane_model)
    
    def cali_t_f_angle(self):
        angle = np.arccos(np.dot(self.front_plane_model_b, self.top_plane_model) / (np.linalg.norm(self.front_plane_model_b) * np.linalg.norm(self.top_plane_model)))
        self.tf_angle_degrees = np.degrees(angle)
        if 0 <= self.tf_angle_degrees <=90:
            self.tf_angle_degrees = 180 - self.tf_angle_degrees

    def find_front_plane_l(self,pts):
        self.front_point_clouds_l = pts.copy()
        front_plane_model , self.front_center_l,self.front_plane_points_l,self.n_front_side_points =  self.find_plane(self.front_point_clouds_l)
        self.front_plane_d_l = front_plane_model[3]
        self.front_plane_model_l = front_plane_model[:3]

    def find_front_plane_r(self,pts):
        self.front_point_clouds_r = pts.copy()
        front_plane_model , self.front_center_r,self.front_plane_points_r,self.n_front_side_points=  self.find_plane(self.front_point_clouds_r)
        self.front_plane_d_r = front_plane_model[3]
        self.front_plane_model_r = front_plane_model[:3]

    def find_left_plane(self,pts):
        left_plane_points = pts.copy()
        left_plane_model , self.left_center,self.left_plane_points ,_ = self.find_plane(left_plane_points ,  threshold=0.0003)
        self.left_plane_d = left_plane_model[3]
        self.left_plane_model = left_plane_model[:3]

    def find_right_plane(self,pts):
        right_point_clouds = pts.copy()
        right_plane_model , self.right_center ,self.right_plane_points,_=  self.find_plane(right_point_clouds,threshold=0.0003)
        self.right_plane_d = right_plane_model[3]
        self.right_plane_model = right_plane_model[:3]

    def find_b_f_line_l(self):
        self.b_f_line_direction_l = np.cross(self.front_plane_model_l, self.bottom_plane_model)

    def cali_b_f_angle_l(self):
        angle = np.arccos(np.dot(self.front_plane_model_l, self.bottom_plane_model) / (np.linalg.norm(self.front_plane_model_l) * np.linalg.norm(self.bottom_plane_model)))
        self.bf_angle_degrees_l = np.degrees(angle)
        if 0 <= self.bf_angle_degrees_l <=90:
            self.bf_angle_degrees_l = 180 - self.bf_angle_degrees_l

    def find_t_f_line_l(self):
        self.t_f_line_direction_l = np.cross(self.front_plane_model_l, self.top_plane_model)

    def cali_t_f_angle_l(self):
        angle = np.arccos(np.dot(self.front_plane_model_l, self.top_plane_model) / (np.linalg.norm(self.front_plane_model_l) * np.linalg.norm(self.top_plane_model)))
        self.tf_angle_degrees_l = np.degrees(angle)
        if 90 <= self.tf_angle_degrees_l <= 180:
            self.tf_angle_degrees_l = 180 - self.tf_angle_degrees_l
    
    def cali_angle(self,plane_1,plane_2):
        angle = np.arccos(np.dot(plane_1, plane_2) / (np.linalg.norm(plane_1) * np.linalg.norm(plane_2)))
        return np.degrees(angle)


    def find_surface_points_l(self):
        pts = self.left_point_clouds.copy()
        _, _,_ ,none_plane_points=  self.find_plane(pts)
        _, _,_ ,self.left_surface_points=  self.find_plane(none_plane_points)
        
    def processing_left_point_clouds(self):
        
        self.find_b_f_line_l()
        self.cali_b_f_angle_l()
        self.find_t_f_line_l()
        self.cali_t_f_angle_l()
        self.find_surface_points_l()
        self.find_left_coordinate()
        self.transform_left_point_cloud()
        self.find_top_plane_in_left_coordinate()
        self.find_bottom_plane_in_left_coordinate()
        
    def find_bottom_plane_in_left_coordinate(self):
        bottom_plane_points_in_left_coordinate = (self.left_coordinate.T @ self.bottom_plane_points.T).T
        bottom_plane_model_left , self.bottom_center_left,_,_ =  self.find_plane(bottom_plane_points_in_left_coordinate )
        self.bottom_plane_model_left = bottom_plane_model_left[:3]

    def find_top_plane_in_left_coordinate(self):
        top_plane_points_in_left_coordinate = (self.left_coordinate.T @ self.top_plane_points.T).T
        top_plane_model_left , self.top_center_left,_,_ =  self.find_plane(top_plane_points_in_left_coordinate )
        self.top_plane_model_left = top_plane_model_left[:3]
        
    def transform_left_point_cloud(self):
        self.transform_left_plane_to_bottom()
        self.transform_front_plane_to_bottom_l()
        self.transform_surface_plane_to_bottom_l()
        self.cali_bottom_coordinate_inv()

    def cali_bottom_coordinate_inv(self):
        self.bottom_coordinate_inv = np.linalg.inv(self.bottom_coordinate)

    def transform_left_plane_to_bottom(self):
        self.left_plane_points_transformed, self.bottom_center_left,self.bottom_coordinate = self.project_points_to_plane(self.left_plane_points,self.bottom_plane_model , self.bottom_center)

    def transform_front_plane_to_bottom_l(self):
        self.front_plane_points_l_transformed, self.bottom_center_left,self.bottom_coordinate = self.project_points_to_plane(self.front_plane_points_l,self.bottom_plane_model , self.bottom_center)

    def transform_surface_plane_to_bottom_l(self):
        self.left_surface_points_transformed, self.bottom_center_left,self.bottom_coordinate = self.project_points_to_plane(self.left_surface_points,self.bottom_plane_model , self.bottom_center)

    def find_left_coordinate(self):
            # 计算两个平面法向量的叉积
        line_direction = np.cross(self.left_plane_model, self.front_plane_model_l)

        rz = line_direction
        rx = np.cross(rz,(1,0,0))
        ry = np.cross(rx,rz)
        rx = rx / np.linalg.norm(rx)
        ry = ry / np.linalg.norm(ry)
        rz = rz / np.linalg.norm(rz)
        R = np.c_[rx, ry, rz]

        self.left_coordinate = R
        self.left_coordinate_inv = np.linalg.inv(R)
        
    def processing_right_point_clouds(self):
        pts = self.right_point_clouds.copy()
        _, _,_ ,none_plane_points=  self.find_plane(pts)
        _, _,_ ,self.right_surface_points=  self.find_plane(none_plane_points)

        line_direction = np.cross(self.right_plane_model, self.front_plane_model_r)

        rz = line_direction
        rx = np.cross(rz,(1,0,0))
        ry = np.cross(rx,rz)
        rx = rx / np.linalg.norm(rx)
        ry = ry / np.linalg.norm(ry)
        rz = rz / np.linalg.norm(rz)
        R = np.c_[rx, ry, rz]

        self.right_coordinate = R
        self.right_coordinate_inv = np.linalg.inv(R)

        self.right_plane_points_transformed, self.bottom_center_right,self.bottom_coordinate = self.project_points_to_plane(self.right_plane_points,self.bottom_plane_model , self.bottom_center)
        self.front_plane_points_r_transformed, self.bottom_center_right,self.bottom_coordinate = self.project_points_to_plane(self.front_plane_points_r,self.bottom_plane_model , self.bottom_center)
        self.right_surface_points_transformed, self.bottom_center_right,self.bottom_coordinate = self.project_points_to_plane(self.right_surface_points,self.bottom_plane_model , self.bottom_center)
        self.bottom_coordinate_inv = np.linalg.inv(self.bottom_coordinate)

        top_plane_points_in_right_coordinate = (self.right_coordinate.T @ self.top_plane_points.T).T
        top_plane_model_right , self.top_center_right,_,_ =  self.find_plane(top_plane_points_in_right_coordinate )
        self.top_plane_model_right = top_plane_model_right[:3]

        bottom_plane_points_in_right_coordinate = (self.right_coordinate.T @ self.bottom_plane_points.T).T
        bottom_plane_model_right , self.bottom_center_right,_,_ =  self.find_plane(bottom_plane_points_in_right_coordinate )
        self.bottom_plane_model_right = bottom_plane_model_right[:3]

        max_z = np.max(self.right_plane_points_transformed[:, 2])
        min_z = np.min(self.right_plane_points_transformed[:, 2])

        # 计算最大z和最小z的中间值
        mid_z = (max_z + min_z) / 2
        distance_to_move = 0.03  # 移动的距离
        self.three_points = [mid_z, mid_z - distance_to_move, mid_z + distance_to_move]  # 移动 point[2]

        self.right_plane_points_in_right_coordinate = (self.right_coordinate.T @ self.right_plane_points.T).T
        self.front_plane_points_in_right_coordinate = (self.right_coordinate.T @ self.front_plane_points_r.T).T
        self.surface_plane_points_in_right_coordinate = (self.right_coordinate.T @ self.right_surface_points.T).T

        self.right_plane_tree = KDTree(self.right_plane_points_in_right_coordinate)
        self.right_surface_plane_tree = KDTree(self.surface_plane_points_in_right_coordinate)
        self.right_front_plane_tree = KDTree(self.front_plane_points_in_right_coordinate)
        
        self.right_plane_matching_points = []
        # 上下浮动
        tolerance = 0.0003
                # 在点云中查找与给定点 z 值相近的点
        for p in self.right_plane_points_transformed:
            if abs(p[2] - self.three_points[0]) <= tolerance or abs(p[2] - self.three_points[1]) <= tolerance or abs(p[2] - self.three_points[2]) <= tolerance :
                 self.right_plane_matching_points.append(p)

        self.front_plane_matching_points_r = []
        # 上下浮动
        tolerance = 0.0003
        # 在点云中查找与给定点 z 值相近的点
        for p in self.front_plane_points_r_transformed:
            if abs(p[2] - self.three_points[0]) <= tolerance or abs(p[2] - self.three_points[1]) <= tolerance or abs(p[2] - self.three_points[2]) <= tolerance :
                 self.front_plane_matching_points_r.append(p)
        # self.front_plane_matching_points_l_tree = KDTree(self.front_plane_matching_points_l)
        self.surface_plane_matching_points_r = []
        # 上下浮动
        tolerance = 0.0002
        # 在点云中查找与给定点 z 值相近的点
        for p in self.right_surface_points_transformed:
            #找到距离引线点最近的点：
            # dist1, idx = self.front_plane_matching_points_l_tree.query(p, k=1)  # 查询最近的一个邻点
            # dist2, idx = self.left_plane_matching_points_tree.query(p, k=1)  # 查询最近的一个邻点
            if abs(p[2] - self.three_points[0]) <= tolerance or abs(p[2] - self.three_points[1]) <= tolerance or abs(p[2] - self.three_points[2]) <= tolerance :
               # if dist1 > 0.002 and dist2 >0.002:
                self.surface_plane_matching_points_r.append(p)
        
        self.right_plane_matching_points_restored = (self.bottom_coordinate_inv.T @ np.array(self.right_plane_matching_points).T).T
        self.right_plane_matching_points_in_right_coordinate = (self.right_coordinate.T @ np.array(self.right_plane_matching_points_restored).T).T
        self.front_plane_matching_points_restored_r = (self.bottom_coordinate_inv.T @ np.array(self.front_plane_matching_points_r).T).T
        self.front_plane_matching_points_in_right_coordinate = (self.right_coordinate.T @ np.array(self.front_plane_matching_points_restored_r).T).T
        self.surface_plane_matching_points_restored_r = (self.bottom_coordinate_inv.T @ np.array(self.surface_plane_matching_points_r).T).T
        self.surface_plane_matching_points_in_right_coordinate = (self.right_coordinate.T @ np.array(self.surface_plane_matching_points_restored_r).T).T

        self.right_plane_top_inter_points = []
        self.right_plane_bottom_inter_points = []
        search_length = 0.02
        for p in self.right_plane_matching_points_in_right_coordinate:
            point = p.copy()
            point[2] = point[2] - search_length

            #找到距离引线点最近的点：
            dist, idx = self.right_plane_tree.query(point, k=1)  # 查询最近的一个邻点
            
            if dist < 0.0002:
                line_direction = point - p
                bottom_t = np.dot(self.bottom_plane_model_right, (self.bottom_center_right - p)) / np.dot(self.bottom_plane_model_right, line_direction)
                bottom_intersection_point = p + bottom_t * line_direction
                self.right_plane_bottom_inter_points.append(bottom_intersection_point)
                top_t = np.dot(self.top_plane_model_right, (self.top_center_right - p)) / np.dot(self.top_plane_model_right, line_direction)
                top_intersection_point = p + top_t * line_direction
                self.right_plane_top_inter_points.append(top_intersection_point)
                point =  (self.right_coordinate_inv.T @ np.array(point).T).T
                p =  (self.right_coordinate_inv.T @ np.array(p).T).T
                #self.v.Line(np.array([point,p]).flatten(),1,[1,0,1])

        if self.right_plane_matching_points_in_right_coordinate[0][2] > self.right_plane_bottom_inter_points[0][2]:
            self.right_plane_top_inter_points = self.right_plane_top_inter_points  - np.array([0,0,0])
            self.right_plane_bottom_inter_points  = self.right_plane_bottom_inter_points +np.array([0,0,0])
        else:
            self.right_plane_top_inter_points = self.right_plane_top_inter_points +np.array([0,0,0])
            self.right_plane_bottom_inter_points = self.right_plane_bottom_inter_points -np.array([0,0,0])

        # if self.right_plane_matching_points_in_right_coordinate[0][2] > self.right_plane_bottom_inter_points[0][2]:
        #     self.right_plane_top_inter_points = self.right_plane_top_inter_points  - np.array([0,0,0.02])
        #     self.right_plane_bottom_inter_points  = self.right_plane_bottom_inter_points +np.array([0,0,0.02])
        # else:
        #     self.right_plane_top_inter_points = self.right_plane_top_inter_points +np.array([0,0,0.02])
        #     self.right_plane_bottom_inter_points = self.right_plane_bottom_inter_points -np.array([0,0,0.02])

        self.right_plane_top_inter_points_restored = (self.right_coordinate_inv.T @ np.array(self.right_plane_top_inter_points).T).T

        # 初始化最大和最小距离及其对应的点
        top_dis = 0.05
        max_distance = -np.inf
        min_distance = np.inf
        self.max_point_on_right_top = None
        self.min_point_on_right_top = None
        for p in self.right_plane_top_inter_points_restored:
            # 计算点到前面平面的距离
            distance = np.abs(np.dot(self.front_plane_model_r,p - self.front_center_r))
            # 更新最大距离及其对应的点
            if distance > max_distance and distance < top_dis:
                max_distance = distance
                self.max_point_on_right_top = p
            
            # 更新最小距离及其对应的点
            if distance < min_distance:
                min_distance = distance
                self.min_point_on_right_top = p

        self.right_plane_bottom_inter_points_restored = (self.right_coordinate_inv.T @ np.array(self.right_plane_bottom_inter_points).T).T
         # 初始化最大和最小距离及其对应的点
        bottom_dis = 0.1
        max_distance = -np.inf
        min_distance = np.inf
        self.max_point_on_right_bottom = None
        self.min_point_on_right_bottom = None
        for p in self.right_plane_bottom_inter_points_restored:
            # 计算点到前面平面的距离
            distance = np.abs(np.dot(self.front_plane_model_r,p - self.front_center_r))
            # 更新最大距离及其对应的点
            if distance > max_distance and distance < bottom_dis:
                max_distance = distance
                self.max_point_on_right_bottom = p
            
            # 更新最小距离及其对应的点
            if distance < min_distance:
                min_distance = distance
                self.min_point_on_right_bottom = p

        self.front_plane_top_inter_points_r = []
        self.front_plane_bottom_inter_points_r = []
        search_length = 0.02
        for p in self.front_plane_matching_points_in_right_coordinate:
            point = p.copy()
            point[2] = point[2] - search_length

                        #找到距离引线点最近的点：
            dist, idx = self.right_front_plane_tree.query(point, k=1)  # 查询最近的一个邻点
            
            if dist < 0.0002:
                line_direction = point - p
                bottom_t = np.dot(self.bottom_plane_model_right, (self.bottom_center_right - p)) / np.dot(self.bottom_plane_model_right, line_direction)
                bottom_intersection_point = p + bottom_t * line_direction
                self.front_plane_bottom_inter_points_r.append(bottom_intersection_point)
                top_t = np.dot(self.top_plane_model_right, (self.top_center_right - p)) / np.dot(self.top_plane_model_right, line_direction)
                top_intersection_point = p + top_t * line_direction
                self.front_plane_top_inter_points_r.append(top_intersection_point)
                point =  (self.right_coordinate_inv.T @ np.array(point).T).T
                p =  (self.right_coordinate_inv.T @ np.array(p).T).T
               # self.v.Line(np.array([point,p]).flatten(),1,[1,0,1])

        # if self.front_plane_matching_points_in_right_coordinate[0][2] > self.front_plane_bottom_inter_points_r[0][2]:
        #     self.front_plane_top_inter_points_r = self.front_plane_top_inter_points_r  - np.array([0,0,0.02])
        #     self.front_plane_bottom_inter_points_r  = self.front_plane_bottom_inter_points_r +np.array([0,0,0.02])
        # else:
        #     self.front_plane_top_inter_points_r = self.front_plane_top_inter_points_r +np.array([0,0,0.02])
        #     self.front_plane_bottom_inter_points_r = self.front_plane_bottom_inter_points_r -np.array([0,0,0.02])

        if self.front_plane_matching_points_in_right_coordinate[0][2] > self.front_plane_bottom_inter_points_r[0][2]:
            self.front_plane_top_inter_points_r = self.front_plane_top_inter_points_r  - np.array([0,0,0])
            self.front_plane_bottom_inter_points_r  = self.front_plane_bottom_inter_points_r +np.array([0,0,0])
        else:
            self.front_plane_top_inter_points_r = self.front_plane_top_inter_points_r +np.array([0,0,0])
            self.front_plane_bottom_inter_points_r = self.front_plane_bottom_inter_points_r -np.array([0,0,0])

        self.front_plane_top_inter_points_r_restored = (self.right_coordinate_inv.T @ np.array(self.front_plane_top_inter_points_r).T).T

            # 初始化最大和最小距离及其对应的点

        max_distance = -np.inf
        min_distance = np.inf
        self.max_point_on_front_top_r = None
        self.min_point_on_front_top_r = None
        for p in self.front_plane_top_inter_points_r_restored:
            # 计算点到前面平面的距离
            distance = np.abs(np.dot(self.right_plane_model,p - self.right_center))
            # 更新最大距离及其对应的点
            if distance > max_distance :
                max_distance = distance
                self.min_point_on_front_top_r = p
            
            # 更新最小距离及其对应的点
            if distance < min_distance:
                min_distance = distance
                self.max_point_on_front_top_r = p

        self.front_plane_bottom_inter_points_r_restored = (self.right_coordinate_inv.T @ np.array(self.front_plane_bottom_inter_points_r).T).T

                # 初始化最大和最小距离及其对应的点

        max_distance = -np.inf
        min_distance = np.inf
        self.max_point_on_front_bottom_r = None
        self.min_point_on_front_bottom_r = None
        for p in self.front_plane_bottom_inter_points_r_restored:
            # 计算点到前面平面的距离
            distance = np.abs(np.dot(self.right_plane_model,p - self.right_center))
            # 更新最大距离及其对应的点
            if distance > max_distance:
                max_distance = distance
                self.min_point_on_front_bottom_r = p
            
            # 更新最小距离及其对应的点
            if distance < min_distance:
                min_distance = distance
                self.max_point_on_front_bottom_r = p

        self.surface_plane_top_inter_points_r = []
        self.surface_plane_bottom_inter_points_r = []
        search_length = 0.02
        for p in self.surface_plane_matching_points_in_right_coordinate:
            point = p.copy()
            point[2] = point[2] - search_length
                        #找到距离引线点最近的点：
            dist, idx = self.right_surface_plane_tree.query(point, k=1)  # 查询最近的一个邻点
            
            if dist <= 0.0005:
                line_direction = point - p
                bottom_t = np.dot(self.bottom_plane_model_right, (self.bottom_center_right - p)) / np.dot(self.bottom_plane_model_right, line_direction)
                bottom_intersection_point = p + bottom_t * line_direction
                self.surface_plane_bottom_inter_points_r.append(bottom_intersection_point)
                top_t = np.dot(self.top_plane_model_right, (self.top_center_right - p)) / np.dot(self.top_plane_model_right, line_direction)
                top_intersection_point = p + top_t * line_direction
                self.surface_plane_top_inter_points_r.append(top_intersection_point)
                point =  (self.right_coordinate_inv.T @ np.array(point).T).T
                p =  (self.right_coordinate_inv.T @ np.array(p).T).T
                #self.v.Line(np.array([point,p]).flatten(),1,[1,0,1])

        # if self.surface_plane_matching_points_in_right_coordinate[0][2] > self.surface_plane_bottom_inter_points_r[0][2]:
        #     self.surface_plane_top_inter_points_r = self.surface_plane_top_inter_points_r  - np.array([0,0,0.02])
        #     self.surface_plane_bottom_inter_points_r  = self.surface_plane_bottom_inter_points_r +np.array([0,0,0.02])
        # else:
        #     self.surface_plane_top_inter_points_r = self.surface_plane_top_inter_points_r +np.array([0,0,0.02])
        #     self.surface_plane_bottom_inter_points_r = self.surface_plane_bottom_inter_points_r -np.array([0,0,0.02])

        if self.surface_plane_matching_points_in_right_coordinate[0][2] > self.surface_plane_bottom_inter_points_r[0][2]:
            self.surface_plane_top_inter_points_r = self.surface_plane_top_inter_points_r  - np.array([0,0,0])
            self.surface_plane_bottom_inter_points_r  = self.surface_plane_bottom_inter_points_r +np.array([0,0,0])
        else:
            self.surface_plane_top_inter_points_r = self.surface_plane_top_inter_points_r +np.array([0,0,0])
            self.surface_plane_bottom_inter_points_r = self.surface_plane_bottom_inter_points_r -np.array([0,0,0])


        self.surface_plane_top_inter_points_r_restored = (self.right_coordinate_inv.T @ np.array(self.surface_plane_top_inter_points_r).T).T

                # 初始化最大和最小距离及其对应的点

        max_distance = np.inf
        min_distance = np.inf
        self.max_point_on_surface_top_r = None
        self.min_point_on_surface_top_r = None
        for p in self.surface_plane_top_inter_points_r_restored:
            # 计算点到前面平面的距离
            distance_min_right = np.linalg.norm(p - self.min_point_on_right_top)
            distance_max_front = np.linalg.norm(p - self.max_point_on_front_top_r)
            # 更新最大距离及其对应的点
            if distance_min_right < max_distance :
                max_distance = distance_min_right
                self.max_point_on_surface_top_r = p
            
            # 更新最小距离及其对应的点
            if distance_max_front < min_distance:
                min_distance = distance_max_front
                self.min_point_on_surface_top_r = p

        self.surface_plane_bottom_inter_points_r_restored = (self.right_coordinate_inv.T @ np.array(self.surface_plane_bottom_inter_points_r).T).T


            # 初始化最大和最小距离及其对应的点

        max_distance = np.inf
        min_distance = np.inf
        self.max_point_on_surface_bottom_r = None
        self.min_point_on_surface_bottom_r = None
        for p in self.surface_plane_bottom_inter_points_r_restored:
            # 计算点到前面平面的距离
            distance_min_right = np.linalg.norm(p - self.min_point_on_right_bottom)
            distance_max_front = np.linalg.norm(p - self.max_point_on_front_bottom_r)
            # 更新最大距离及其对应的点
            if distance_min_right < max_distance :
                max_distance = distance_min_right
                self.max_point_on_surface_bottom_r = p
            
            # 更新最小距离及其对应的点
            if distance_max_front < min_distance:
                min_distance = distance_max_front
                self.min_point_on_surface_bottom_r = p
            # 计算两个点之间的向量

        vector = self.min_point_on_right_top - self.max_point_on_right_top

        # 计算线段的总长度
        length = np.linalg.norm(vector)

        # 计算需要等分的段数
        num_segments = int(length / 0.003)
       # num_segments = math.ceil(length / 0.003)

        # 计算每个分段的向量
        segment_vector = vector / num_segments

        # 计算等分点的坐标
        self.right_top_line_points = [self.max_point_on_right_top + i * segment_vector for i in range(0, num_segments+1)]

        top_plane_model = self.top_plane_model.copy()
        right_plane_model = self.right_plane_model.copy()
        if np.dot(self.top_plane_model,[0,0,1]) > 0:
            top_plane_model = self.top_plane_model * -1
        if np.dot(self.right_plane_model,[0,0,1]) <0:
            right_plane_model = self.right_plane_model * -1
        #rz = (right_plane_model + top_plane_model)/2
        ry = top_plane_model
        ry = ry / np.linalg.norm(ry)
        
        rx = self.min_point_on_right_top - self.max_point_on_right_top
        rz = np.cross(rx,ry)

        rx =rx / np.linalg.norm(rx)
        rz =rz / np.linalg.norm(rz)
        vec1 = rz.copy()
        
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
        q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree = np.rad2deg(pose_radius)


        rz = right_plane_model
        rz = rz / np.linalg.norm(rz)
        vec2 = rz.copy()
        rx = self.min_point_on_right_top - self.max_point_on_right_top
        ry = np.cross(rx,rz)

        rx =rx / np.linalg.norm(rx)
        ry =ry / np.linalg.norm(ry)
        
        
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
        #q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree_new = np.rad2deg(pose_radius)

        vec_angle = self.cali_angle(vec1,vec2)
        print("左上线")
        print(vec_angle)
        self.right_top_line_degree = pose_degree.copy()
        self.right_top_line_quaternion = q
        angle = self.cali_angle(self.right_plane_model,self.top_plane_model)
        if angle > 90:
            angle = 180 -angle

        self.right_plane_top_angle = angle.copy()
        new_array_path = np.array([[vec1[0], vec1[1], vec1[2],
                                    vec2[0],vec2[1],vec2[2]]])
        self.right_top_line_path = np.concatenate((np.array(self.right_top_line_points),
                         new_array_path.repeat(np.array(self.right_top_line_points).shape[0], axis=0)), axis=1)
        new_array_dis = np.array([[q[1],q[2], q[3],q[0]]])
        self.right_top_line_path_dis =  np.concatenate((np.array(self.right_top_line_points),
                         new_array_dis.repeat(np.array(self.right_top_line_points).shape[0], axis=0)), axis=1)
        
        x_coords = np.array(self.right_top_line_path)[:, 0]

        # # 对x坐标进行排序，得到排序后的索引
        sorted_indices = np.argsort(x_coords)[::-1]

        # # 根据排序后的索引重新排列点云
        self.right_top_line_path = np.array(self.right_top_line_path)[sorted_indices]
        

        for p in self.right_top_line_path_dis:
            self.v.Axes(p[:3],p[3:],0.002,3)


        vector = self.min_point_on_right_bottom - self.max_point_on_right_bottom

        # 计算线段的总长度
        length = np.linalg.norm(vector)

        # 计算需要等分的段数
        num_segments = int(length / 0.003)

        # 计算每个分段的向量
        segment_vector = vector / num_segments

        # 计算等分点的坐标
        self.right_bottom_line_points = [self.max_point_on_right_bottom + i * segment_vector for i in range(0, num_segments+1)]
        #rz = (self.left_plane_model + self.top_plane_model_left)/2
        #rz = self.left_plane_model
        bottom_plane_model = self.bottom_plane_model.copy()
        right_plane_model = self.right_plane_model.copy()
        if np.dot(self.bottom_plane_model,[0,0,1]) > 0:
            bottom_plane_model = self.bottom_plane_model * -1
        if np.dot(self.right_plane_model,[0,0,1]) <0:
            right_plane_model = self.right_plane_model * -1

        ry = bottom_plane_model
        ry = ry / np.linalg.norm(ry)
        rx = self.min_point_on_right_bottom - self.max_point_on_right_bottom
        rz = np.cross(rx,ry)
        rx =rx / np.linalg.norm(rx)
        rz =rz / np.linalg.norm(rz)
        vec1 =rz.copy()
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
        q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree = np.rad2deg(pose_radius)



        rz = right_plane_model
        rz = rz / np.linalg.norm(rz)
        vec2 =rz.copy()
        rx = self.min_point_on_right_bottom - self.max_point_on_right_bottom
        ry = np.cross(rx,rz)

        rx =rx / np.linalg.norm(rx)
        ry =ry / np.linalg.norm(ry)
        
        
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
        #q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree_new = np.rad2deg(pose_radius)

        vec_angle = self.cali_angle(vec1,vec2)
        print("左下线")
        print(vec_angle)
        self.right_bottom_line_degree = pose_degree.copy()
        self.right_bottom_line_quaternion = q
        angle = self.cali_angle(self.right_plane_model,self.bottom_plane_model)
        if angle > 90:
            angle = 180 -angle

        self.right_plane_bottom_angle = angle.copy()
        new_array_path = np.array([[vec1[0], vec1[1], vec1[2],
                                    vec2[0],vec2[1],vec2[2]]])
        self.right_bottom_line_path = np.concatenate((np.array(self.right_bottom_line_points),
                         new_array_path.repeat(np.array(self.right_bottom_line_points).shape[0], axis=0)), axis=1)
        new_array_dis = np.array([[q[1],q[2], q[3],q[0]]])
        self.right_bottom_line_path_dis =  np.concatenate((np.array(self.right_bottom_line_points),
                         new_array_dis.repeat(np.array(self.right_bottom_line_points).shape[0], axis=0)), axis=1)

        for p in self.right_bottom_line_path_dis:
            self.v.Axes(p[:3],p[3:],0.002,3)


        self.right_top_curve_points = []
        for p in self.surface_plane_top_inter_points_r_restored:
            if  self.min_point_on_surface_top_r[0] >= p[0] >=self.max_point_on_surface_top_r[0]:
                self.right_top_curve_points.append(p)

        
        model , center,_,_=self.find_plane(np.array(self.right_top_curve_points))
        points, points_center,coordinate = self.project_points_to_plane(np.array(self.right_top_curve_points),model , center)
        # 提取数据点
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # 计算曲线的长度
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        total_distance = np.sum(distances)

        # 拟合多项式曲线
        degree = 8  # 多项式的阶数
        coefficients = np.polyfit(x, y, degree)

        # 使用更密集的点进行拟合
        x_fit = np.linspace(min(x), max(x), 1000)
        x_fit = np.linspace(min(x), max(x), 1000)
        y_fit = np.polyval(coefficients, x_fit)

        # 计算起点到终点的总长度
        total_distance = np.sqrt((x_fit[-1]-x_fit[0])**2 + (y_fit[-1]-y_fit[0])**2)

        # 计算需要等分的段数
        num_segments = int(total_distance / 0.003) + 1

        # 计算每个段的长度
        segment_length = total_distance / num_segments

        # 添加起点的坐标
        x_sampled = [x_fit[0]]
        y_sampled = [y_fit[0]]
        z_sampled = [np.interp(x_fit[0], x, z)]

        # 在拟合曲线上等分取点
        current_distance = 0
        for i in range(len(x_fit)-1):
            distance = np.sqrt((x_fit[i+1]-x_fit[i])**2 + (y_fit[i+1]-y_fit[i])**2)
            current_distance += distance
            if current_distance >= segment_length:
                x_sampled.append(x_fit[i+1])
                y_sampled.append(y_fit[i+1])
                z_sampled.append(np.interp(x_fit[i+1], x, z))
                current_distance = 0

        # 判断终点与最后一个添加的采样点之间的距离
        distance_to_last_sample = np.sqrt((x_fit[-1]-x_sampled[-1])**2 + (y_fit[-1]-y_sampled[-1])**2)
        if distance_to_last_sample >= segment_length/2:
            # 添加终点的坐标
            x_sampled.append(x_fit[-1])
            y_sampled.append(y_fit[-1])
            z_sampled.append(np.interp(x_fit[-1], x, z))

        # plt.scatter(x, y, label='原始数据')
        # plt.plot(x_fit, y_fit, 'r', label='拟合曲线')
        # plt.scatter(x_sampled, y_sampled, color='g', label='采样点')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.legend()

        # plt.show()


        # 将采样点的坐标放在一起
        sampled_points = np.column_stack((x_sampled, y_sampled, z_sampled))
        coordinate_inv = np.linalg.inv(coordinate)
        self.right_top_curve_points = (coordinate_inv.T @ sampled_points.T).T

        new_right_top_curve_points = []
        for p in self.right_top_curve_points:
                if np.linalg.norm(self.min_point_on_surface_top_r - p)>=0.002 and np.linalg.norm(self.max_point_on_surface_top_r -p)>=0.002:
                    new_right_top_curve_points.append(p)
          #  if  self.min_point_on_surface_top_r[0]-0.002 >= p[0] >=self.max_point_on_surface_top_r[0]+0.002:
                    # new_right_top_curve_points.append(p)

        self.right_top_curve_points = new_right_top_curve_points


        #p = Utils.UniformDownSample(pts=np.array( self.right_top_curve_points),k=3)
        # p = Utils.VoxelDownSample(pts=np.array( self.right_top_curve_points),voxel_size=0.002)
        # # p = Utils.VoxelDownSample(pts=p,voxel_size=0.003)
        # #p = Utils.VoxelDownSample(pts=p,voxel_size=0.003)
        # self.right_top_curve_points = Utils.VoxelDownSample(pts=p,voxel_size=0.003)
        self.right_top_curve_points_in_right_coordinate =  (self.right_coordinate.T @ np.array(self.right_top_curve_points).T).T
        self.other_right_top_curve_points_in_right_coordinate = self.right_top_curve_points_in_right_coordinate + np.array([0,0,0.01])
        self.right_top_curve_points_restored =  (self.right_coordinate_inv.T @ np.array(self.right_top_curve_points_in_right_coordinate).T).T
        self.other_right_top_curve_points_restored =  (self.right_coordinate_inv.T @ np.array(self.other_right_top_curve_points_in_right_coordinate).T).T


        self.right_top_curve_path =[]
        self.right_top_curve_path_dis = []
        x_coords = np.array(self.right_top_curve_points_restored)[:, 0]

        # # 对x坐标进行排序，得到排序后的索引
        sorted_indices = np.argsort(x_coords)[::-1]

        # # 根据排序后的索引重新排列点云
        self.right_top_curve_points_restored = np.array(self.right_top_curve_points_restored)[sorted_indices]
        self.other_right_top_curve_points_restored = np.array(self.other_right_top_curve_points_restored)[sorted_indices]

        top_plane_model = self.top_plane_model.copy()
        for p in range(len(self.right_top_curve_points_restored)):
            
            if p == len(self.right_top_curve_points_restored) - 1:
                # p是最后一个元素
                p1 = self.right_top_curve_points_restored[p-1]
                p2 = self.right_top_curve_points_restored[p]
                p3 = self.other_right_top_curve_points_restored[p]
                # 在这里处理最后一个元素的情况
            # elif p == len(self.right_top_curve_points_restored) - 2:
            #     p1 = self.right_top_curve_points_restored[p-2]
            #     p2 = self.right_top_curve_points_restored[p]
            #     p3 = self.other_right_top_curve_points_restored[p]
            else:
                p1 = self.right_top_curve_points_restored[p]
                p2 = self.right_top_curve_points_restored[p+1]
                p3 = self.other_right_top_curve_points_restored[p]

                # 在这里处理p和p+1的情况
            plane1 = np.array([p1,p2,p3])
            plane1_model ,_,_,_ = self.find_plane(plane1)

            plane1_model = plane1_model[:3]
            if np.dot(self.top_plane_model,[0,0,1]) > 0:
                top_plane_model = self.top_plane_model * -1
            if np.dot(plane1_model,[0,0,1]) <0:
                plane1_model = plane1_model * -1

            ry = top_plane_model
            ry = ry / np.linalg.norm(ry)
            
            rx = p1 -p2
            rz = np.cross(rx,ry)
            
            rx =rx / np.linalg.norm(rx)
            rz =rz / np.linalg.norm(rz)
            vec1 = rz.copy()
            T = np.eye(4)
            T[:3, 0] = rx
            T[:3, 1] = ry
            T[:3, 2] = rz
            q = transformations.quaternion_from_matrix(T) 
            pose_radius = transformations.euler_from_matrix(T)
            pose_degree = np.rad2deg(pose_radius)

            rz = plane1_model
            rz = rz / np.linalg.norm(rz)
            vec2 = rz.copy()
            rx = p1 -p2
            ry = np.cross(rx,rz)

            rx =rx / np.linalg.norm(rx)
            ry =ry / np.linalg.norm(ry)
            
            
            T = np.eye(4)
            T[:3, 0] = rx
            T[:3, 1] = ry
            T[:3, 2] = rz
            #q = transformations.quaternion_from_matrix(T) 
            self.T = T
            pose_radius = transformations.euler_from_matrix(T)
            pose_degree_new = np.rad2deg(pose_radius)

            vec_angle = self.cali_angle(vec1,vec2)
            print("右上曲线")
            print(vec_angle)
            curve_p0_degree = pose_degree.copy()
            curve_p0_quaternion = q
            angle = self.cali_angle(plane1_model,top_plane_model)
            if angle > 90:
                angle = 180 -angle

            curve_p0_degree = angle.copy()
            curve_p0 = np.array([self.right_top_curve_points_restored[p][0],self.right_top_curve_points_restored[p][1],self.right_top_curve_points_restored[p][2],
                                vec1[0],vec1[1],vec1[2],vec2[0],
                                vec2[1],vec2[2] ])
            curve_p0_dis = np.array([self.right_top_curve_points_restored[p][0],self.right_top_curve_points_restored[p][1],self.right_top_curve_points_restored[p][2],
                                q[1],q[2], q[3],q[0] ])
            self.right_top_curve_path.append(curve_p0)
            self.right_top_curve_path_dis.append(curve_p0_dis)


            for p in self.right_top_curve_path_dis:
                self.v.Axes(p[:3],p[3:],0.002,3)

            # self.left_top_curve_segmentation()
        self.right_bottom_curve_points = []
        for p in self.surface_plane_bottom_inter_points_r_restored:
            if  self.min_point_on_surface_bottom_r[0]  >= p[0] >=self.max_point_on_surface_bottom_r[0]:
                self.right_bottom_curve_points.append(p)

        model , center,_,_=self.find_plane(np.array(self.right_bottom_curve_points))
        points, points_center,coordinate = self.project_points_to_plane(np.array(self.right_bottom_curve_points),model , center)
        # 提取数据点
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # 计算曲线的长度
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        total_distance = np.sum(distances)

        # 拟合多项式曲线
        degree = 8  # 多项式的阶数
        coefficients = np.polyfit(x, y, degree)

        # 使用更密集的点进行拟合
        x_fit = np.linspace(min(x), max(x), 1000)
        y_fit = np.polyval(coefficients, x_fit)

        # 计算起点到终点的总长度
        total_distance = np.sqrt((x_fit[-1]-x_fit[0])**2 + (y_fit[-1]-y_fit[0])**2)

        # 计算需要等分的段数
        num_segments = int(total_distance / 0.003) + 1

        # 计算每个段的长度
        segment_length = total_distance / num_segments

        # 添加起点的坐标
        x_sampled = [x_fit[0]]
        y_sampled = [y_fit[0]]
        z_sampled = [np.interp(x_fit[0], x, z)]

        # 在拟合曲线上等分取点
        current_distance = 0
        for i in range(len(x_fit)-1):
            distance = np.sqrt((x_fit[i+1]-x_fit[i])**2 + (y_fit[i+1]-y_fit[i])**2)
            current_distance += distance
            if current_distance >= segment_length:
                x_sampled.append(x_fit[i+1])
                y_sampled.append(y_fit[i+1])
                z_sampled.append(np.interp(x_fit[i+1], x, z))
                current_distance = 0

        # 判断终点与最后一个添加的采样点之间的距离
        distance_to_last_sample = np.sqrt((x_fit[-1]-x_sampled[-1])**2 + (y_fit[-1]-y_sampled[-1])**2)
        if distance_to_last_sample >= segment_length/2:
            # 添加终点的坐标
            x_sampled.append(x_fit[-1])
            y_sampled.append(y_fit[-1])
            z_sampled.append(np.interp(x_fit[-1], x, z))


        # plt.scatter(x, y, label='原始数据')
        # plt.plot(x_fit, y_fit, 'r', label='拟合曲线')
        # plt.scatter(x_sampled, y_sampled, color='g', label='采样点')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.legend()

        # plt.show()


        # 将采样点的坐标放在一起
        sampled_points = np.column_stack((x_sampled, y_sampled, z_sampled))
        coordinate_inv = np.linalg.inv(coordinate)
        self.right_bottom_curve_points = (coordinate_inv.T @ sampled_points.T).T
        new_right_bottom_curve_points = []
        for p in self.right_bottom_curve_points:
           # if self.min_point_on_surface_bottom_r[0]-0.002  >= p[0] >=self.max_point_on_surface_bottom_r[0]+0.002:
            if np.linalg.norm(self.min_point_on_surface_bottom_r -p)>=0.002 and np.linalg.norm(self.max_point_on_surface_bottom_r-p)>=0.002:

                new_right_bottom_curve_points.append(p)


        self.right_bottom_curve_points = new_right_bottom_curve_points
        

        # p = Utils.VoxelDownSample(pts=np.array( self.right_bottom_curve_points),voxel_size=0.003)
        # p = Utils.VoxelDownSample(pts=p,voxel_size=0.003)
        # self.right_bottom_curve_points = Utils.VoxelDownSample(pts=p,voxel_size=0.003)
     
        # self.transform_left_top_curve_to_left_coordinate() 
        self.right_bottom_curve_points_in_right_coordinate =  (self.right_coordinate.T @ np.array(self.right_bottom_curve_points).T).T

        # self.find_left_top_curve_other_point_on_line()
        self.other_right_bottom_curve_points_in_right_coordinate = self.right_bottom_curve_points_in_right_coordinate + np.array([0,0,0.01])

        # self.restore_left_top_curve_points()
        self.right_bottom_curve_points_restored =  (self.right_coordinate_inv.T @ np.array(self.right_bottom_curve_points_in_right_coordinate).T).T
 
        # self.restore_other_left_top_curve_points()
        self.other_right_bottom_curve_points_restored =  (self.right_coordinate_inv.T @ np.array(self.other_right_bottom_curve_points_in_right_coordinate).T).T

        # self.find_left_top_curve_path()

        self.right_bottom_curve_path =[]
        self.right_bottom_curve_path_dis = []
        x_coords = np.array(self.right_bottom_curve_points_restored)[:, 0]

        # # 对x坐标进行排序，得到排序后的索引
        sorted_indices = np.argsort(x_coords)[::-1]

        # # 根据排序后的索引重新排列点云
        self.right_bottom_curve_points_restored = np.array(self.right_bottom_curve_points_restored)[sorted_indices]
        self.other_right_bottom_curve_points_restored = np.array(self.other_right_bottom_curve_points_restored)[sorted_indices]

        bottom_plane_model = self.bottom_plane_model.copy()
                
        for p in range(len(self.right_bottom_curve_points_restored)):
            
            if p == len(self.right_bottom_curve_points_restored) - 1:
                # p是最后一个元素
                p1 = self.right_bottom_curve_points_restored[p-1]
                p2 = self.right_bottom_curve_points_restored[p]
                p3 = self.other_right_bottom_curve_points_restored[p]
                # 在这里处理最后一个元素的情况
            # elif p == len(self.right_bottom_curve_points_restored) - 2:
            #     # p是最后一个元素
            #     p1 = self.right_bottom_curve_points_restored[p-2]
            #     p2 = self.right_bottom_curve_points_restored[p]
            #     p3 = self.other_right_bottom_curve_points_restored[p]
            else:
                p1 = self.right_bottom_curve_points_restored[p]
                p2 = self.right_bottom_curve_points_restored[p+1]
                p3 = self.other_right_bottom_curve_points_restored[p]
                # 在这里处理p和p+1的情况
            plane1 = np.array([p1,p2,p3])
            plane1_model ,_,_,_ = self.find_plane(plane1)

            plane1_model = plane1_model[:3]
            if np.dot(self.bottom_plane_model,[0,0,1]) > 0:
                bottom_plane_model = self.bottom_plane_model * -1
            if np.dot(plane1_model,[0,0,1]) <0:
                plane1_model = plane1_model * -1

            ry = bottom_plane_model
            ry = ry / np.linalg.norm(ry)
            
            rx = p1 - p2
            rz = np.cross(rx,ry)

            rx =rx / np.linalg.norm(rx)
            rz =rz / np.linalg.norm(rz)
            vec1 = rz.copy()
            
            
            T = np.eye(4)
            T[:3, 0] = rx
            T[:3, 1] = ry
            T[:3, 2] = rz
            q = transformations.quaternion_from_matrix(T) 
            pose_radius = transformations.euler_from_matrix(T)
            pose_degree = np.rad2deg(pose_radius)


            rz = plane1_model
            rz = rz / np.linalg.norm(rz)
            vec2 = rz.copy()
            rx = p1 -p2
            ry = np.cross(rx,rz)

            rx =rx / np.linalg.norm(rx)
            ry =ry / np.linalg.norm(ry)
            
            
            T = np.eye(4)
            T[:3, 0] = rx
            T[:3, 1] = ry
            T[:3, 2] = rz
            #q = transformations.quaternion_from_matrix(T) 
            self.T = T
            pose_radius = transformations.euler_from_matrix(T)
            pose_degree_new = np.rad2deg(pose_radius)

            vec_angle = self.cali_angle(vec1,vec2)
            print("右下曲线")
            print(vec_angle)
            curve_p0_degree = pose_degree.copy()
            curve_p0_quaternion = q
            angle = self.cali_angle(plane1_model,bottom_plane_model)
            if angle > 90:
                angle = 180 -angle

            curve_p0_degree = angle.copy()
            curve_p0 = np.array([self.right_bottom_curve_points_restored[p][0],self.right_bottom_curve_points_restored[p][1],self.right_bottom_curve_points_restored[p][2],
                                vec1[0],vec1[1],vec1[2],
                                 vec2[0],vec2[1],vec2[2]])
            curve_p0_dis = np.array([self.right_bottom_curve_points_restored[p][0],self.right_bottom_curve_points_restored[p][1],self.right_bottom_curve_points_restored[p][2],
                                q[1],q[2], q[3],q[0] ])
            self.right_bottom_curve_path.append(curve_p0)
            self.right_bottom_curve_path_dis.append(curve_p0_dis)
            for p in self.right_bottom_curve_path_dis:
                self.v.Axes(p[:3],p[3:],0.002,3)





        # 计算两个点之间的向量
        vector = self.max_point_on_front_bottom_l - self.max_point_on_front_bottom_r

        # 计算线段的总长度
        length = np.linalg.norm(vector)

        # 计算需要等分的段数
        num_segments = int(length / 0.003)

        # 计算每个分段的向量
        segment_vector = vector / num_segments

        # 计算等分点的坐标
        self.front_bottom_line_points = [self.max_point_on_front_bottom_r + i * segment_vector for i in range(0, num_segments+1)]


        bottom_plane_model = self.bottom_plane_model.copy()
        front_plane_model = self.front_plane_model_r.copy()

        if np.dot(self.bottom_plane_model,[0,0,1]) > 0:
            bottom_plane_model = self.bottom_plane_model * -1
        if np.dot(front_plane_model,[0,0,1]) <0:
            front_plane_model = self.front_plane_model * -1

        ry = bottom_plane_model
        ry = ry / np.linalg.norm(ry)
        
        rx = self.min_point_on_front_bottom_r - self.max_point_on_front_bottom_r
        rz = np.cross(rx,ry)

        rx =rx / np.linalg.norm(rx)
        rz =rz / np.linalg.norm(rz)
        vec1 = rz.copy()
        
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
        q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree = np.rad2deg(pose_radius)

        rz = front_plane_model
        rz = rz / np.linalg.norm(rz)
        vec2 = rz.copy()
        rx = self.min_point_on_front_bottom_r - self.max_point_on_front_bottom_r
        ry = np.cross(rx,rz)

        rx =rx / np.linalg.norm(rx)
        ry =ry / np.linalg.norm(ry)
        
        
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
        #q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree_new = np.rad2deg(pose_radius)
        

        self.right_front_line_degree = pose_degree.copy()
        self.right_front_line_quaternion = q
        angle = self.cali_angle(front_plane_model,self.bottom_plane_model)
        vec_angle = self.cali_angle(vec1,vec2)
        print("下前线")
        print(vec_angle)
        if angle > 90:
            angle = 180 -angle

        self.front_plane_bottom_angle = angle.copy()
        new_array_path = np.array([[vec1[0], vec1[1], vec1[2],
                                    vec2[0],vec2[1],vec2[2]]])
        self.front_bottom_line_path = np.concatenate((np.array(self.front_bottom_line_points),
                         new_array_path.repeat(np.array(self.front_bottom_line_points).shape[0], axis=0)), axis=1)
        new_array_dis = np.array([[q[1],q[2], q[3],q[0]]])
        self.front_bottom_line_path_dis =  np.concatenate((np.array(self.front_bottom_line_points),
                         new_array_dis.repeat(np.array(self.front_bottom_line_points).shape[0], axis=0)), axis=1)
        for p in self.front_bottom_line_path_dis:
            self.v.Axes(p[:3],p[3:],0.002,3)



            # 计算两个点之间的向量
        # vector = self.max_point_on_front_top_r - self.max_point_on_front_top_r
        vector = self.max_point_on_front_top_l - self.max_point_on_front_top_r
        # 计算线段的总长度
        length = np.linalg.norm(vector)

        # 计算需要等分的段数
        num_segments = int(length / 0.003)

        # 计算每个分段的向量
        segment_vector = vector / num_segments

        # 计算等分点的坐标
        self.front_top_line_points = [self.max_point_on_front_top_r + i * segment_vector for i in range(0, num_segments+1)]

        top_plane_model = self.top_plane_model.copy()
        front_plane_model = self.front_plane_model_r.copy()

        if np.dot(self.top_plane_model,[0,0,1]) > 0:
            top_plane_model = self.top_plane_model * -1
        if np.dot(front_plane_model,[0,0,1]) <0:
            front_plane_model = self.front_plane_model * -1

        ry = top_plane_model
        ry = ry / np.linalg.norm(ry)
        
        rx = self.min_point_on_front_top_r - self.max_point_on_front_top_r
        rz = np.cross(rx,ry)

        rx =rx / np.linalg.norm(rx)
        rz =rz / np.linalg.norm(rz)
        vec1 = rz.copy()
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
        q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree = np.rad2deg(pose_radius)


        rz = front_plane_model
        rz = rz / np.linalg.norm(rz)
        vec2 = rz.copy()
        rx = self.min_point_on_front_top_r - self.max_point_on_front_top_r
        ry = np.cross(rx,rz)

        rx =rx / np.linalg.norm(rx)
        ry =ry / np.linalg.norm(ry)
        
        
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
       # q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree_new = np.rad2deg(pose_radius)

        vec_angle = self.cali_angle(vec1,vec2)
        print("右上前线")
        print(vec_angle)
        self.right_front_line_degree = pose_degree.copy()
        self.right_front_line_quaternion = q
        angle = self.cali_angle(front_plane_model,self.top_plane_model)
        if angle > 90:
            angle = 180 -angle

        self.front_plane_top_angle = angle.copy()
        new_array_path = np.array([[vec1[0], vec1[1], vec1[2],
                                    vec2[0],vec2[1],vec2[2]]])
        self.front_top_line_path = np.concatenate((np.array(self.front_top_line_points),
                         new_array_path.repeat(np.array(self.front_top_line_points).shape[0], axis=0)), axis=1)
        new_array_dis = np.array([[q[1],q[2], q[3],q[0]]])
        self.front_top_line_path_dis =  np.concatenate((np.array(self.front_top_line_points),
                         new_array_dis.repeat(np.array(self.front_top_line_points).shape[0], axis=0)), axis=1)

                # # 对x坐标进行排序，得到排序后的索引
        x_coords = self.front_top_line_path[:, 0]
        sorted_indices = np.argsort(x_coords)[::-1]

        # # 根据排序后的索引重新排列点云
        self.front_top_line_path = np.array(self.front_top_line_path)[sorted_indices]
        for p in self.front_top_line_path_dis:
            self.v.Axes(p[:3],p[3:],0.002,3)



    def find_left_center_points(self,tolerance):
        self.matching_points = []
        # 找到和左侧中心点相近的地方
        bottom_inv_R = np.linalg.inv(self.bottom_R)
        point = self.left_center.copy()
        point = self.bottom_R.T @ point

        # 在点云中查找与给定点 z 值相近的点
        for p in self.tmp_point_clouds:
            if abs(p[2] - point[2]) <= tolerance:
                self.matching_points.append(p)

        self.matching_points = (bottom_inv_R.T @ np.array(self.matching_points).T).T

        self.matching_points = (self.best_R.T @ np.array(self.matching_points).T).T

    def find_right_center_points(self,tolerance):
        # 工件中心一条点云
        self.matching_points = []
        bottom_inv_R = np.linalg.inv(self.bottom_R)
        point = self.right_center.copy()
        point = self.bottom_R.T @ point

        distance_to_move = 0.03  # 移动的距离
        distance_threshold = 0.0001  # 判断是否非常接近的阈值

        moved_point = [point[2], point[2] - distance_to_move, point[2] + distance_to_move]  # 移动 point[2]

        # 在点云中查找与给定点 z 值相近的点
        for p in self.tmp_point_clouds:
            if abs(p[2] - moved_point[0]) <= tolerance or abs(p[2] - moved_point[1]) <= tolerance or abs(p[2] - moved_point[2]) <= tolerance :
                 self.matching_points.append(p)
        min_distance = 0.003

        self.matching_points = (bottom_inv_R.T @ np.array(self.matching_points).T).T

        self.matching_points = (self.best_R.T @ np.array(self.matching_points).T).T

        self.r_matching_points = []
        for p in self.right_plane_pts:
            if abs(p[2] - moved_point[0]) <= tolerance or abs(p[2] - moved_point[1]) <= tolerance or abs(p[2] - moved_point[2]) <= tolerance :
                 self.r_matching_points.append(p)

        self.r_matching_points = (bottom_inv_R.T @ np.array(self.r_matching_points).T).T

        self.r_matching_points = (self.best_R.T @ np.array(self.r_matching_points).T).T

        #在右侧平面上找到和底面交线，然后将右侧平面上引线和底面交点中，找到距离前面中点最近的点和最远的点
        # 最近的点就是，交线中靠近我们的点，最远的就是远离我们的点，最远的应该需要裁剪之后去找
   

    def cut_right_welding_points(self,top_dis = 0.05,bottom_dis = 0.1):      
        # 距离前面100mm的距离，定义起点终点
        # 距离点云距离 6mm的 位置防止左右重叠
        point_cloud = np.array(self.bottom_intersection_points).copy()
        
        for p in self.bottom_new_intersection_points:
            dis = cdist(point_cloud, np.array([p]), metric='euclidean')
            min_distance = np.min(dis)
            distance = np.abs(np.dot(self.front_plane_model,p - self.front_center))
            if distance < bottom_dis and min_distance > 0.006:
                self.bottom_intersection_points.append(p)

        #self.top_intersection_points = []
        point_cloud = np.array(self.top_intersection_points).copy()
        for p in self.top_new_intersection_points:
            dis = cdist(point_cloud, np.array([p]), metric='euclidean')
            min_distance = np.min(dis)
            distance = np.abs(np.dot(self.front_plane_model,p - self.front_center))
            if distance < top_dis and min_distance > 0.006:
                self.top_intersection_points.append(p)

                #embed()

    def cut_left_welding_points(self,top_dis = 0.05,bottom_dis = 0.1):
        self.bottom_intersection_points = []
        for p in self.bottom_new_intersection_points:
            distance = np.abs(np.dot(self.front_plane_model,p - self.front_center))
            if distance < bottom_dis:
                self.bottom_intersection_points.append(p)

        self.top_intersection_points = []
        for p in self.top_new_intersection_points:
            distance = np.abs(np.dot(self.front_plane_model,p - self.front_center))
            if distance < top_dis:
                self.top_intersection_points.append(p)


        # # 打印排序后的点云
        # print(sorted_points)

    def down_sample_right(self):
        #如果我想确保每隔3mm一个点呢
   
        self.bottom_intersection_points = Utils.VoxelDownSample(pts= np.array(self.bottom_intersection_points),voxel_size=0.003)
        self.bottom_intersection_points = Utils.VoxelDownSample(pts= np.array(self.bottom_intersection_points),voxel_size=0.003)
        # self.top_intersection_points = Utils.UniformDownSample(pts= np.array(self.top_intersection_points),k =3)
        # self.bottom_intersection_points = Utils.UniformDownSample(pts= np.array(self.bottom_intersection_points),k = 3)
        # self.top_intersection_points = Utils.VoxelDownSample(pts= np.array(self.top_intersection_points),voxel_size=0.003)
        self.bottom_intersection_points = Utils.VoxelDownSample(pts= np.array(self.bottom_intersection_points),voxel_size=0.003)


    def remove_duplicate_points(self,point_cloud, threshold):
        num_points = point_cloud.shape[0]
        keep_indices = np.ones(num_points, dtype=bool)

        for i in range(num_points):
            if keep_indices[i]:
                current_point = point_cloud[i]

                # Find points with higher Y values
                higher_y_indices = np.where((point_cloud[:, 1] > current_point[1]) & keep_indices)[0]

                # Mark points with lower Y values for removal
                keep_indices[higher_y_indices] = False

        # Keep only the points that are marked for keeping
        filtered_point_cloud = point_cloud[keep_indices]

        return filtered_point_cloud

    def down_sample_left(self):
        self.top_intersection_points = Utils.UniformDownSample(pts= np.array(self.top_intersection_points),k =3)
        self.bottom_intersection_points = Utils.UniformDownSample(pts= np.array(self.bottom_intersection_points),k =3)
        self.top_intersection_points = Utils.VoxelDownSample(pts= np.array(self.top_intersection_points),voxel_size=0.003)
        self.bottom_intersection_points = Utils.VoxelDownSample(pts= np.array(self.bottom_intersection_points),voxel_size=0.003)

    def find_three_points_from_left_plane(self):
        # 找到最大z和最小z的值
        max_z = np.max(self.left_plane_points_transformed[:, 2])
        min_z = np.min(self.left_plane_points_transformed[:, 2])

        # 计算最大z和最小z的中间值
        mid_z = (max_z + min_z) / 2
        distance_to_move = 0.03  # 移动的距离
        self.three_points = [mid_z, mid_z - distance_to_move, mid_z + distance_to_move]  # 移动 point[2]

    def find_point_group_in_left_plane(self):
        self.left_plane_matching_points = []
        # 上下浮动
        tolerance = 0.0003
                # 在点云中查找与给定点 z 值相近的点
        for p in self.left_plane_points_transformed:
            if abs(p[2] - self.three_points[0]) <= tolerance or abs(p[2] - self.three_points[1]) <= tolerance or abs(p[2] - self.three_points[2]) <= tolerance :
                 self.left_plane_matching_points.append(p)

        # self.left_plane_matching_points_tree = KDTree(self.left_plane_matching_points)

    def find_point_group_in_front_plane_l(self):
        self.front_plane_matching_points_l = []
        # 上下浮动
        tolerance = 0.0003
        # 在点云中查找与给定点 z 值相近的点
        for p in self.front_plane_points_l_transformed:
            if abs(p[2] - self.three_points[0]) <= tolerance or abs(p[2] - self.three_points[1]) <= tolerance or abs(p[2] - self.three_points[2]) <= tolerance :
                 self.front_plane_matching_points_l.append(p)
        # self.front_plane_matching_points_l_tree = KDTree(self.front_plane_matching_points_l)

    def find_point_group_in_surface_plane_l(self):
        self.surface_plane_matching_points_l = []
        # 上下浮动
        tolerance = 0.0003
        # 在点云中查找与给定点 z 值相近的点
        for p in self.left_surface_points_transformed:
            #找到距离引线点最近的点：
            # dist1, idx = self.front_plane_matching_points_l_tree.query(p, k=1)  # 查询最近的一个邻点
            # dist2, idx = self.left_plane_matching_points_tree.query(p, k=1)  # 查询最近的一个邻点
            if abs(p[2] - self.three_points[0]) <= tolerance or abs(p[2] - self.three_points[1]) <= tolerance or abs(p[2] - self.three_points[2]) <= tolerance :
               # if dist1 > 0.002 and dist2 >0.002:
                self.surface_plane_matching_points_l.append(p)


    def find_inter_points_from_left_plane_group(self):
        
        self.left_plane_top_inter_points = []
        self.left_plane_bottom_inter_points = []
        search_length = 0.02
        for p in self.left_plane_matching_points_in_left_coordinate:
            point = p.copy()
            point[2] = point[2] - search_length

            #找到距离引线点最近的点：
            dist, idx = self.left_plane_tree.query(point, k=1)  # 查询最近的一个邻点
            
            if dist < 0.0002:
                line_direction = point - p
                bottom_t = np.dot(self.bottom_plane_model_left, (self.bottom_center_left - p)) / np.dot(self.bottom_plane_model_left, line_direction)
                bottom_intersection_point = p + bottom_t * line_direction
                self.left_plane_bottom_inter_points.append(bottom_intersection_point)
                top_t = np.dot(self.top_plane_model_left, (self.top_center_left - p)) / np.dot(self.top_plane_model_left, line_direction)
                top_intersection_point = p + top_t * line_direction
                self.left_plane_top_inter_points.append(top_intersection_point)
                point =  (self.left_coordinate_inv.T @ np.array(point).T).T
                p =  (self.left_coordinate_inv.T @ np.array(p).T).T
                #self.v.Line(np.array([point,p]).flatten(),1,[1,0,1])

        # if self.left_plane_matching_points_in_left_coordinate[0][2] > self.left_plane_bottom_inter_points[0][2]:
        #     self.left_plane_top_inter_points = self.left_plane_top_inter_points  - np.array([0,0,0.02])
        #     self.left_plane_bottom_inter_points  = self.left_plane_bottom_inter_points +np.array([0,0,0.02])
        # else:
        #     self.left_plane_top_inter_points = self.left_plane_top_inter_points +np.array([0,0,0.02])
        #     self.left_plane_bottom_inter_points = self.left_plane_bottom_inter_points -np.array([0,0,0.02])

        if self.left_plane_matching_points_in_left_coordinate[0][2] > self.left_plane_bottom_inter_points[0][2]:
            self.left_plane_top_inter_points = self.left_plane_top_inter_points  - np.array([0,0,0])
            self.left_plane_bottom_inter_points  = self.left_plane_bottom_inter_points +np.array([0,0,0])
        else:
            self.left_plane_top_inter_points = self.left_plane_top_inter_points +np.array([0,0,0])
            self.left_plane_bottom_inter_points = self.left_plane_bottom_inter_points -np.array([0,0,0])


    def restore_left_plane_top_group(self):
        self.left_plane_top_inter_points_restored = (self.left_coordinate_inv.T @ np.array(self.left_plane_top_inter_points).T).T

    def restore_left_plane_bottom_group(self):
        self.left_plane_bottom_inter_points_restored = (self.left_coordinate_inv.T @ np.array(self.left_plane_bottom_inter_points).T).T


    def find_inter_points_from_front_plane_group_l(self):
        
        self.front_plane_top_inter_points_l = []
        self.front_plane_bottom_inter_points_l = []
        search_length = 0.02
        for p in self.front_plane_matching_points_in_left_coordinate:
            point = p.copy()
            point[2] = point[2] - search_length

                        #找到距离引线点最近的点：
            dist, idx = self.left_front_plane_tree.query(point, k=1)  # 查询最近的一个邻点
            
            if dist < 0.0002:
                line_direction = point - p
                bottom_t = np.dot(self.bottom_plane_model_left, (self.bottom_center_left - p)) / np.dot(self.bottom_plane_model_left, line_direction)
                bottom_intersection_point = p + bottom_t * line_direction
                self.front_plane_bottom_inter_points_l.append(bottom_intersection_point)
                top_t = np.dot(self.top_plane_model_left, (self.top_center_left - p)) / np.dot(self.top_plane_model_left, line_direction)
                top_intersection_point = p + top_t * line_direction
                self.front_plane_top_inter_points_l.append(top_intersection_point)
                point =  (self.left_coordinate_inv.T @ np.array(point).T).T
                p =  (self.left_coordinate_inv.T @ np.array(p).T).T
               # self.v.Line(np.array([point,p]).flatten(),1,[1,0,1])

        # if self.front_plane_matching_points_in_left_coordinate[0][2] > self.front_plane_bottom_inter_points_l[0][2]:
        #     self.front_plane_top_inter_points_l = self.front_plane_top_inter_points_l  - np.array([0,0,0.02])
        #     self.front_plane_bottom_inter_points_l  = self.front_plane_bottom_inter_points_l +np.array([0,0,0.02])
        # else:
        #     self.front_plane_top_inter_points_l = self.front_plane_top_inter_points_l +np.array([0,0,0.02])
        #     self.front_plane_bottom_inter_points_l = self.front_plane_bottom_inter_points_l -np.array([0,0,0.02])

        if self.front_plane_matching_points_in_left_coordinate[0][2] > self.front_plane_bottom_inter_points_l[0][2]:
            self.front_plane_top_inter_points_l = self.front_plane_top_inter_points_l  - np.array([0,0,0])
            self.front_plane_bottom_inter_points_l  = self.front_plane_bottom_inter_points_l +np.array([0,0,0])
        else:
            self.front_plane_top_inter_points_l = self.front_plane_top_inter_points_l +np.array([0,0,0])
            self.front_plane_bottom_inter_points_l = self.front_plane_bottom_inter_points_l -np.array([0,0,0])

    def restore_front_plane_top_group_l(self):
        self.front_plane_top_inter_points_l_restored = (self.left_coordinate_inv.T @ np.array(self.front_plane_top_inter_points_l).T).T

    def restore_front_plane_bottom_group_l(self):
        self.front_plane_bottom_inter_points_l_restored = (self.left_coordinate_inv.T @ np.array(self.front_plane_bottom_inter_points_l).T).T

    def find_inter_points_from_surface_plane_group_l(self):
        
        self.surface_plane_top_inter_points_l = []
        self.surface_plane_bottom_inter_points_l = []
        search_length = 0.02
        for p in self.surface_plane_matching_points_in_left_coordinate:
            point = p.copy()
            point[2] = point[2] - search_length
                        #找到距离引线点最近的点：
            dist, idx = self.left_surface_plane_tree.query(point, k=1)  # 查询最近的一个邻点
            
            if dist <= 0.001:
                line_direction = point - p
                bottom_t = np.dot(self.bottom_plane_model_left, (self.bottom_center_left - p)) / np.dot(self.bottom_plane_model_left, line_direction)
                bottom_intersection_point = p + bottom_t * line_direction
                self.surface_plane_bottom_inter_points_l.append(bottom_intersection_point)
                top_t = np.dot(self.top_plane_model_left, (self.top_center_left - p)) / np.dot(self.top_plane_model_left, line_direction)
                top_intersection_point = p + top_t * line_direction
                self.surface_plane_top_inter_points_l.append(top_intersection_point)
                point =  (self.left_coordinate_inv.T @ np.array(point).T).T
                p =  (self.left_coordinate_inv.T @ np.array(p).T).T
               # self.v.Line(np.array([point,p]).flatten(),1,[1,0,1])

        # if self.surface_plane_matching_points_in_left_coordinate[0][2] > self.surface_plane_bottom_inter_points_l[0][2]:
        #     self.surface_plane_top_inter_points_l = self.surface_plane_top_inter_points_l  - np.array([0,0,0.02])
        #     self.surface_plane_bottom_inter_points_l  = self.surface_plane_bottom_inter_points_l +np.array([0,0,0.02])
        # else:
        #     self.surface_plane_top_inter_points_l = self.surface_plane_top_inter_points_l +np.array([0,0,0.02])
        #     self.surface_plane_bottom_inter_points_l = self.surface_plane_bottom_inter_points_l -np.array([0,0,0.02])


        if self.surface_plane_matching_points_in_left_coordinate[0][2] > self.surface_plane_bottom_inter_points_l[0][2]:
            self.surface_plane_top_inter_points_l = self.surface_plane_top_inter_points_l  - np.array([0,0,0])
            self.surface_plane_bottom_inter_points_l  = self.surface_plane_bottom_inter_points_l +np.array([0,0,0])
        else:
            self.surface_plane_top_inter_points_l = self.surface_plane_top_inter_points_l +np.array([0,0,0])
            self.surface_plane_bottom_inter_points_l = self.surface_plane_bottom_inter_points_l -np.array([0,0,0])
            

    def restore_surface_plane_top_group_l(self):
        self.surface_plane_top_inter_points_l_restored = (self.left_coordinate_inv.T @ np.array(self.surface_plane_top_inter_points_l).T).T

    def restore_surface_plane_bottom_group_l(self):
        self.surface_plane_bottom_inter_points_l_restored = (self.left_coordinate_inv.T @ np.array(self.surface_plane_bottom_inter_points_l).T).T

    def restore_left_plane_point_group(self):
        self.left_plane_matching_points_restored = (self.bottom_coordinate_inv.T @ np.array(self.left_plane_matching_points).T).T

    def transform_left_plane_point_group_to_left_coordinate(self):
        self.left_plane_matching_points_in_left_coordinate = (self.left_coordinate.T @ np.array(self.left_plane_matching_points_restored).T).T

    def restore_front_plane_point_group_l(self):
        self.front_plane_matching_points_restored_l = (self.bottom_coordinate_inv.T @ np.array(self.front_plane_matching_points_l).T).T

    def transform_front_plane_point_group_to_left_coordinate_l(self):
        self.front_plane_matching_points_in_left_coordinate = (self.left_coordinate.T @ np.array(self.front_plane_matching_points_restored_l).T).T

    def restore_left_surface_plane_point_group(self):
        self.surface_plane_matching_points_restored_l = (self.bottom_coordinate_inv.T @ np.array(self.surface_plane_matching_points_l).T).T

    def transform_left_surface_plane_point_group_to_left_coordinate(self):
        self.surface_plane_matching_points_in_left_coordinate = (self.left_coordinate.T @ np.array(self.surface_plane_matching_points_restored_l).T).T

    def find_left_bottom_line_path(self):
        # self.left_top_line_segmentation()
                # 计算两个点之间的向量
        vector = self.min_point_on_left_bottom - self.max_point_on_left_bottom

        # 计算线段的总长度
        length = np.linalg.norm(vector)

        # 计算需要等分的段数
        num_segments = int(length / 0.003)

        # 计算每个分段的向量
        segment_vector = vector / num_segments

        # 计算等分点的坐标
        self.left_bottom_line_points = [self.max_point_on_left_bottom + i * segment_vector for i in range(0, num_segments+1)]
        #rz = (self.left_plane_model + self.top_plane_model_left)/2
        #rz = self.left_plane_model
        bottom_plane_model = self.bottom_plane_model.copy()
        left_plane_model = self.left_plane_model.copy()
        if np.dot(self.bottom_plane_model,[0,0,1]) > 0:
            bottom_plane_model = self.bottom_plane_model * -1
        if np.dot(self.left_plane_model,[0,0,1]) <0:
            left_plane_model = self.left_plane_model * -1

        ry = bottom_plane_model
        ry = ry / np.linalg.norm(ry)
        rx = self.max_point_on_left_bottom - self.min_point_on_left_bottom
        rz = np.cross(rx,ry)
        
        rx =rx / np.linalg.norm(rx)
        rz =rz / np.linalg.norm(rz)
        vec1 = rz.copy()
        
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
        q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree = np.rad2deg(pose_radius)



        rz = left_plane_model
        rz = rz / np.linalg.norm(rz)
        vec2 = rz.copy()
        rx = self.max_point_on_left_bottom - self.min_point_on_left_bottom
        ry = np.cross(rx,rz)

        rx =rx / np.linalg.norm(rx)
        ry =ry / np.linalg.norm(ry)
        
        
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
        
        #q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree_new = np.rad2deg(pose_radius)
        vec_angle = self.cali_angle(vec1,vec2)
        print("左下线")
        print(vec_angle)
        self.left_bottom_line_degree = pose_degree.copy()
        self.left_bottom_line_quaternion = q
        angle = self.cali_angle(self.left_plane_model,self.bottom_plane_model)
        if angle > 90:
            angle = 180 -angle
        
        # z_dir = left_plane_model/ np.linalg.norm(left_plane_model) *0.0655
        # self.left_bottom_line_points = self.left_bottom_line_points + z_dir
        self.left_plane_bottom_angle = angle.copy()
        new_array_path = np.array([[vec1[0], vec1[1], vec1[2],
                                    vec2[0],vec2[1],vec2[2]]])
        self.left_bottom_line_path = np.concatenate((np.array(self.left_bottom_line_points),
                         new_array_path.repeat(np.array(self.left_bottom_line_points).shape[0], axis=0)), axis=1)
        new_array_dis = np.array([[q[1],q[2], q[3],q[0]]])
        self.left_bottom_line_path_dis =  np.concatenate((np.array(self.left_bottom_line_points),
                         new_array_dis.repeat(np.array(self.left_bottom_line_points).shape[0], axis=0)), axis=1)
        
        for p in self.left_bottom_line_path_dis:
            self.v.Axes(p[:3],p[3:],0.002,3)

    def find_left_front_path(self):
        # 计算两个点之间的向量
        vector = self.min_point_on_front_bottom_l - self.max_point_on_front_bottom_l
        # 计算线段的总长度
        length = np.linalg.norm(vector)

        # 计算需要等分的段数
        num_segments = int(length / 0.003)

        # 计算每个分段的向量
        segment_vector = vector / num_segments

        # 计算等分点的坐标
        self.front_bottom_line_points = [self.max_point_on_front_bottom_l + i * segment_vector for i in range(0, num_segments+1)]

        bottom_plane_model = self.bottom_plane_model.copy()
        front_plane_model = self.front_plane_model_l.copy()

        if np.dot(self.bottom_plane_model,[0,0,1]) > 0:
            bottom_plane_model = self.bottom_plane_model * -1
        if np.dot(front_plane_model,[0,0,1]) <0:
            front_plane_model = self.front_plane_model * -1

        ry = bottom_plane_model
        ry = ry / np.linalg.norm(ry)
        rx = self.max_point_on_front_bottom_l - self.min_point_on_front_bottom_l
        rz = np.cross(rx,ry)

        rx =rx / np.linalg.norm(rx)
        rz =rz / np.linalg.norm(rz)
        vec1 = rz.copy()
        
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
        q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree = np.rad2deg(pose_radius)


        rz = front_plane_model
        rz = rz / np.linalg.norm(rz)
        vec2 =rz.copy()
        rx = self.max_point_on_front_bottom_l - self.min_point_on_front_bottom_l
        ry = np.cross(rx,rz)

        rx =rx / np.linalg.norm(rx)
        ry =ry / np.linalg.norm(ry)
        
        
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
        #q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree_new = np.rad2deg(pose_radius)
        vec_angle = self.cali_angle(vec1,vec2)
        print("前下线")
        print(vec_angle)
        self.left_front_line_degree = pose_degree.copy()
        self.left_front_line_quaternion = q
        angle = self.cali_angle(front_plane_model,self.bottom_plane_model)
        if angle > 90:
            angle = 180 -angle

        self.front_plane_bottom_angle = angle.copy()
        new_array_path = np.array([[vec1[0], vec1[1], vec1[2],
        vec2[0],vec2[1],vec2[2]]])
        self.front_bottom_line_path = np.concatenate((np.array(self.front_bottom_line_points),
                         new_array_path.repeat(np.array(self.front_bottom_line_points).shape[0], axis=0)), axis=1)
        new_array_dis = np.array([[q[1],q[2], q[3],q[0]]])
        self.front_bottom_line_path_dis =  np.concatenate((np.array(self.front_bottom_line_points),
                         new_array_dis.repeat(np.array(self.front_bottom_line_points).shape[0], axis=0)), axis=1)
        
        for p in self.front_bottom_line_path_dis:
            self.v.Axes(p[:3],p[3:],0.002,3)



            # 计算两个点之间的向量
        vector = self.min_point_on_front_top_l - self.max_point_on_front_top_l
        # 计算线段的总长度
        length = np.linalg.norm(vector)

        # 计算需要等分的段数
        num_segments = int(length / 0.003)

        # 计算每个分段的向量
        segment_vector = vector / num_segments

        # 计算等分点的坐标
        self.front_top_line_points = [self.max_point_on_front_top_l + i * segment_vector for i in range(0, num_segments+1)]

        top_plane_model = self.top_plane_model.copy()
        front_plane_model = self.front_plane_model_l.copy()

        if np.dot(self.top_plane_model,[0,0,1]) > 0:
            top_plane_model = self.top_plane_model * -1
        if np.dot(front_plane_model,[0,0,1]) <0:
            front_plane_model = self.front_plane_model * -1

        ry = top_plane_model
        ry = ry / np.linalg.norm(ry)
        rx = self.max_point_on_front_top_l - self.min_point_on_front_top_l
        rz = np.cross(rx,ry)

        rx =rx / np.linalg.norm(rx)
        rz =rz / np.linalg.norm(rz)
        vec1 = rz.copy()
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
        q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree = np.rad2deg(pose_radius)

        rz = front_plane_model
        rz = rz / np.linalg.norm(rz)
        vec2 = rz.copy()
        rx = self.max_point_on_front_top_l - self.min_point_on_front_top_l
        ry = np.cross(rx,rz)

        rx =rx / np.linalg.norm(rx)
        ry =ry / np.linalg.norm(ry)
        
        
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
       # q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree_new = np.rad2deg(pose_radius)
        self.left_front_line_degree = pose_degree.copy()
        self.left_front_line_quaternion = q
        angle = self.cali_angle(front_plane_model,self.top_plane_model)
        if angle > 90:
            angle = 180 -angle
        vec_angle = self.cali_angle(vec1,vec2)
        print("前下线")
        print(vec_angle)
        self.front_plane_top_angle = angle.copy()
        new_array_path = np.array([[vec1[0], vec1[1], vec1[2],
        vec2[0],vec2[1],vec2[2]]])
        self.front_top_line_path = np.concatenate((np.array(self.front_top_line_points),
                         new_array_path.repeat(np.array(self.front_top_line_points).shape[0], axis=0)), axis=1)
        new_array_dis = np.array([[q[1],q[2], q[3],q[0]]])
        self.front_top_line_path_dis =  np.concatenate((np.array(self.front_top_line_points),
                         new_array_dis.repeat(np.array(self.front_top_line_points).shape[0], axis=0)), axis=1)

        for p in self.front_top_line_path_dis:
            self.v.Axes(p[:3],p[3:],0.002,3)


    def left_top_line_segmentation(self):

        # 计算两个点之间的向量
        vector = self.min_point_on_left_top - self.max_point_on_left_top

        # 计算线段的总长度
        length = np.linalg.norm(vector)

        # 计算需要等分的段数
        num_segments = int(length / 0.003)

        # 计算每个分段的向量
        segment_vector = vector / num_segments

        # 计算等分点的坐标
        self.left_top_line_points = [self.max_point_on_left_top + i * segment_vector for i in range(0, num_segments+1)]

        # 输出等分点的坐标
        # for point in equal_points:
        #     print("等分点坐标:", point)

    def left_top_curve_segmentation(self):
        self.left_top_curve_points = []
        for p in self.surface_plane_top_inter_points_l_restored:
            if  self.min_point_on_surface_top_l[0]  <= p[0] <=self.max_point_on_surface_top_l[0]:
                self.left_top_curve_points.append(p)

        model , center,_,_=self.find_plane(np.array(self.left_top_curve_points))
        points, points_center,coordinate = self.project_points_to_plane(np.array(self.left_top_curve_points),model , center)
        # 提取数据点
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # 计算曲线的长度
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        total_distance = np.sum(distances)

        # 拟合多项式曲线
        degree = 8  # 多项式的阶数
        coefficients = np.polyfit(x, y, degree)

        # 使用更密集的点进行拟合
        x_fit = np.linspace(min(x), max(x), 1000)
        y_fit = np.polyval(coefficients, x_fit)

        # 计算起点到终点的总长度
        total_distance = np.sqrt((x_fit[-1]-x_fit[0])**2 + (y_fit[-1]-y_fit[0])**2)

        # 计算需要等分的段数
        num_segments = int(total_distance / 0.003) + 1

        # 计算每个段的长度
        segment_length = total_distance / num_segments

        # 添加起点的坐标
        x_sampled = [x_fit[0]]
        y_sampled = [y_fit[0]]
        z_sampled = [np.interp(x_fit[0], x, z)]

        # 在拟合曲线上等分取点
        current_distance = 0
        for i in range(len(x_fit)-1):
            distance = np.sqrt((x_fit[i+1]-x_fit[i])**2 + (y_fit[i+1]-y_fit[i])**2)
            current_distance += distance
            if current_distance >= segment_length:
                x_sampled.append(x_fit[i+1])
                y_sampled.append(y_fit[i+1])
                z_sampled.append(np.interp(x_fit[i+1], x, z))
                current_distance = 0

        # 判断终点与最后一个添加的采样点之间的距离
        distance_to_last_sample = np.sqrt((x_fit[-1]-x_sampled[-1])**2 + (y_fit[-1]-y_sampled[-1])**2)
        if distance_to_last_sample >= segment_length/2:
            # 添加终点的坐标
            x_sampled.append(x_fit[-1])
            y_sampled.append(y_fit[-1])
            z_sampled.append(np.interp(x_fit[-1], x, z))

        # plt.scatter(x, y, label='原始数据')
        # plt.plot(x_fit, y_fit, 'r', label='拟合曲线')
        # plt.scatter(x_sampled, y_sampled, color='g', label='采样点')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.legend()

        # plt.show()

        # 将采样点的坐标放在一起
        sampled_points = np.column_stack((x_sampled, y_sampled, z_sampled))
        coordinate_inv = np.linalg.inv(coordinate)
        self.left_top_curve_points = (coordinate_inv.T @ sampled_points.T).T

        new_left_top_curve_points = []
        for p in self.left_top_curve_points:
            if np.linalg.norm(p-self.min_point_on_surface_top_l) >= 0.002 and np.linalg.norm(p-self.max_point_on_surface_top_l) >= 0.002:
                new_left_top_curve_points.append(p)

        self.left_top_curve_points = new_left_top_curve_points
        
       # embed()
        # p = Utils.VoxelDownSample(pts=np.array( self.left_top_curve_points),voxel_size=0.003)
        # p = Utils.VoxelDownSample(pts=p,voxel_size=0.003)
        # self.left_top_curve_points = Utils.VoxelDownSample(pts=p,voxel_size=0.003)
        
    def transform_left_top_line_to_left_coordinate(self):
        self.left_top_line_points_in_left_coordinate =  (self.left_coordinate.T @ np.array(self.left_top_line_points).T).T
    
    def transform_left_top_curve_to_left_coordinate(self):
        self.left_top_curve_points_in_left_coordinate =  (self.left_coordinate.T @ np.array(self.left_top_curve_points).T).T




        # self.transform_left_top_curve_to_left_coordinate() 
        # self.find_left_top_curve_other_point_on_line()
        # self.restore_left_top_curve_points()
        # self.restore_other_left_top_curve_points()
        # self.find_left_top_curve_path()

    def find_left_top_line_point_on_line(self):
        self.other_left_top_line_points_in_left_coordinate = self.left_top_line_points_in_left_coordinate + np.array([0,0,0.01])
    
    def find_left_top_curve_other_point_on_line(self):
        self.other_left_top_curve_points_in_left_coordinate = self.left_top_curve_points_in_left_coordinate + np.array([0,0,0.01])

    def restore_left_top_line_points(self):
        self.left_top_line_points_restored =  (self.left_coordinate_inv.T @ np.array(self.left_top_line_points_in_left_coordinate).T).T

    def restore_left_top_curve_points(self):
        self.left_top_curve_points_restored =  (self.left_coordinate_inv.T @ np.array(self.left_top_curve_points_in_left_coordinate).T).T
    
    def restore_other_left_top_line_points(self):
        self.other_left_top_line_points_restored =  (self.left_coordinate_inv.T @ np.array(self.other_left_top_line_points_in_left_coordinate).T).T
    
    def restore_other_left_top_curve_points(self):
        self.other_left_top_curve_points_restored =  (self.left_coordinate_inv.T @ np.array(self.other_left_top_curve_points_in_left_coordinate).T).T

    def find_left_bottom_curve_path(self):
        # self.left_top_curve_segmentation()
        self.left_bottom_curve_points = []
        for p in self.surface_plane_bottom_inter_points_l_restored:
            if  self.min_point_on_surface_bottom_l[0]  <= p[0] <=self.max_point_on_surface_bottom_l[0]:
                self.left_bottom_curve_points.append(p)

        model , center,_,_=self.find_plane(np.array(self.left_bottom_curve_points))
        points, points_center,coordinate = self.project_points_to_plane(np.array(self.left_bottom_curve_points),model , center)
        # 提取数据点
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # 计算曲线的长度
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        total_distance = np.sum(distances)

        # 拟合多项式曲线
        degree = 8  # 多项式的阶数
        coefficients = np.polyfit(x, y, degree)

        # 使用更密集的点进行拟合
        x_fit = np.linspace(min(x), max(x), 1000)
        y_fit = np.polyval(coefficients, x_fit)

        # 计算起点到终点的总长度
        total_distance = np.sqrt((x_fit[-1]-x_fit[0])**2 + (y_fit[-1]-y_fit[0])**2)

        # 计算需要等分的段数
        num_segments = int(total_distance / 0.003) + 1

        # 计算每个段的长度
        segment_length = total_distance / num_segments

        # 添加起点的坐标
        x_sampled = [x_fit[0]]
        y_sampled = [y_fit[0]]
        z_sampled = [np.interp(x_fit[0], x, z)]

        # 在拟合曲线上等分取点
        current_distance = 0
        for i in range(len(x_fit)-1):
            distance = np.sqrt((x_fit[i+1]-x_fit[i])**2 + (y_fit[i+1]-y_fit[i])**2)
            current_distance += distance
            if current_distance >= segment_length:
                x_sampled.append(x_fit[i+1])
                y_sampled.append(y_fit[i+1])
                z_sampled.append(np.interp(x_fit[i+1], x, z))
                current_distance = 0

        # 判断终点与最后一个添加的采样点之间的距离
        distance_to_last_sample = np.sqrt((x_fit[-1]-x_sampled[-1])**2 + (y_fit[-1]-y_sampled[-1])**2)
        if distance_to_last_sample >= segment_length/2:
            # 添加终点的坐标
            x_sampled.append(x_fit[-1])
            y_sampled.append(y_fit[-1])
            z_sampled.append(np.interp(x_fit[-1], x, z))

        # plt.scatter(x, y, label='原始数据')
        # plt.plot(x_fit, y_fit, 'r', label='拟合曲线')
        # plt.scatter(x_sampled, y_sampled, color='g', label='采样点')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.legend()

        # plt.show()


        # 将采样点的坐标放在一起
        sampled_points = np.column_stack((x_sampled, y_sampled, z_sampled))
        coordinate_inv = np.linalg.inv(coordinate)
        self.left_bottom_curve_points = (coordinate_inv.T @ sampled_points.T).T

        new_left_bottom_curve_points = []
        for p in self.left_bottom_curve_points:
           # if self.min_point_on_surface_bottom_l[0]-0.002  <= p[0] <=self.max_point_on_surface_bottom_l[0]+0.002:
             if np.linalg.norm(self.min_point_on_surface_bottom_l-p) >=0.002 and np.linalg.norm(self.max_point_on_surface_bottom_l - p)>=0.002:
                new_left_bottom_curve_points.append(p)

        self.left_bottom_curve_points = new_left_bottom_curve_points
        

        # p = Utils.VoxelDownSample(pts=np.array( self.left_bottom_curve_points),voxel_size=0.003)
        # p = Utils.VoxelDownSample(pts=p,voxel_size=0.003)
        # self.left_bottom_curve_points = Utils.VoxelDownSample(pts=p,voxel_size=0.003)
     
        # self.transform_left_top_curve_to_left_coordinate() sort
        self.left_bottom_curve_points_in_left_coordinate =  (self.left_coordinate.T @ np.array(self.left_bottom_curve_points).T).T

        # self.find_left_top_curve_other_point_on_line()
        self.other_left_bottom_curve_points_in_left_coordinate = self.left_bottom_curve_points_in_left_coordinate + np.array([0,0,0.01])

        # self.restore_left_top_curve_points()
        self.left_bottom_curve_points_restored =  (self.left_coordinate_inv.T @ np.array(self.left_bottom_curve_points_in_left_coordinate).T).T
 
        # self.restore_other_left_top_curve_points()
        self.other_left_bottom_curve_points_restored =  (self.left_coordinate_inv.T @ np.array(self.other_left_bottom_curve_points_in_left_coordinate).T).T

        # self.find_left_top_curve_path()

        self.left_bottom_curve_path =[]
        self.left_bottom_curve_path_dis = []
        x_coords = np.array(self.left_bottom_curve_points_restored)[:, 0]

        # # 对x坐标进行排序，得到排序后的索引
        sorted_indices = np.argsort(x_coords)[::-1]

        # # 根据排序后的索引重新排列点云
        self.left_bottom_curve_points_restored = np.array(self.left_bottom_curve_points_restored)[sorted_indices]
        self.other_left_bottom_curve_points_restored = np.array(self.other_left_bottom_curve_points_restored)[sorted_indices]

        bottom_plane_model = self.bottom_plane_model.copy()
                
        for p in range(len(self.left_bottom_curve_points_restored)):
            
            if p == len(self.left_bottom_curve_points_restored) - 1:
                # p是最后一个元素
                p1 = self.left_bottom_curve_points_restored[p-1]
                p2 = self.left_bottom_curve_points_restored[p]
                p3 = self.other_left_bottom_curve_points_restored[p]
                # 在这里处理最后一个元素的情况
            # elif p == len(self.left_bottom_curve_points_restored) - 2:
            #     # p是最后一个元素
            #     p1 = self.left_bottom_curve_points_restored[p-2]
            #     p2 = self.left_bottom_curve_points_restored[p]
            #     p3 = self.other_left_bottom_curve_points_restored[p]
                # 在这里处理最后一个元素的情况
            else:
                p1 = self.left_bottom_curve_points_restored[p]
                p2 = self.left_bottom_curve_points_restored[p+1]
                p3 = self.other_left_bottom_curve_points_restored[p]
                # 在这里处理p和p+1的情况
            plane1 = np.array([p1,p2,p3])
            plane1_model ,_,_,_ = self.find_plane(plane1)

            plane1_model = plane1_model[:3]
            if np.dot(self.bottom_plane_model,[0,0,1]) > 0:
                bottom_plane_model = self.bottom_plane_model * -1
            if np.dot(plane1_model,[0,0,1]) <0:
                plane1_model = plane1_model * -1

            ry = bottom_plane_model
            ry = ry / np.linalg.norm(ry)
            rx = p1 - p2
            rz = np.cross(rx,ry)

            rx =rx / np.linalg.norm(rx)
            rz =rz / np.linalg.norm(rz)
            vec1 = rz.copy()
            T = np.eye(4)
            T[:3, 0] = rx
            T[:3, 1] = ry
            T[:3, 2] = rz
            q = transformations.quaternion_from_matrix(T) 
            pose_radius = transformations.euler_from_matrix(T)
            pose_degree = np.rad2deg(pose_radius)


            rz = plane1_model
            rz = rz / np.linalg.norm(rz)
            vec2 = rz.copy()
            rx = p1 - p2
            ry = np.cross(rx,rz)

            rx =rx / np.linalg.norm(rx)
            ry =ry / np.linalg.norm(ry)
            
            
            T = np.eye(4)
            T[:3, 0] = rx
            T[:3, 1] = ry
            T[:3, 2] = rz
           # q = transformations.quaternion_from_matrix(T) 
            self.T = T
            pose_radius = transformations.euler_from_matrix(T)
            pose_degree_new = np.rad2deg(pose_radius)

            curve_p0_degree = pose_degree.copy()
            curve_p0_quaternion = q
            angle = self.cali_angle(plane1_model,bottom_plane_model)
            if angle > 90:
                angle = 180 -angle
            vec_angle = self.cali_angle(vec1,vec2)
            print("左下曲线")
            print(vec_angle)
            curve_p0_degree = angle.copy()
            curve_p0 = np.array([self.left_bottom_curve_points_restored[p][0],self.left_bottom_curve_points_restored[p][1],self.left_bottom_curve_points_restored[p][2],
                                vec1[0],vec1[1],vec1[2],
                                vec2[0],vec2[1],vec2[2] ])
            curve_p0_dis = np.array([self.left_bottom_curve_points_restored[p][0],self.left_bottom_curve_points_restored[p][1],self.left_bottom_curve_points_restored[p][2],
                                q[1],q[2], q[3],q[0] ])
            self.left_bottom_curve_path.append(curve_p0)
            self.left_bottom_curve_path_dis.append(curve_p0_dis)

        for p in self.left_bottom_curve_path_dis:
            self.v.Axes(p[:3],p[3:],0.002,3)


    def find_left_top_curve_path(self):
        self.left_top_curve_path =[]
        self.left_top_curve_path_dis = []
        x_coords = np.array(self.left_top_curve_points_restored)[:, 0]

        # # 对x坐标进行排序，得到排序后的索引
        sorted_indices = np.argsort(x_coords)[::-1]

        # # 根据排序后的索引重新排列点云
        self.left_top_curve_points_restored = np.array(self.left_top_curve_points_restored)[sorted_indices]
        self.other_left_top_curve_points_restored = np.array(self.other_left_top_curve_points_restored)[sorted_indices]

        top_plane_model = self.top_plane_model.copy()
        for p in range(len(self.left_top_curve_points_restored)):
            
            if p == len(self.left_top_curve_points_restored) - 1:
                # p是最后一个元素
                p1 = self.left_top_curve_points_restored[p-1]
                p2 = self.left_top_curve_points_restored[p]
                p3 = self.other_left_top_curve_points_restored[p]
                # 在这里处理最后一个元素的情况
            # elif p == len(self.left_top_curve_points_restored) - 2:
            #     # p是最后一个元素
            #     p1 = self.left_top_curve_points_restored[p-2]
            #     p2 = self.left_top_curve_points_restored[p]
            #     p3 = self.other_left_top_curve_points_restored[p]
            else:
                p1 = self.left_top_curve_points_restored[p]
                p2 = self.left_top_curve_points_restored[p+1]
                p3 = self.other_left_top_curve_points_restored[p]
                # 在这里处理p和p+1的情况
            plane1 = np.array([p1,p2,p3])
            plane1_model ,_,_,_ = self.find_plane(plane1)

            plane1_model = plane1_model[:3]
            if np.dot(self.top_plane_model,[0,0,1]) > 0:
                top_plane_model = self.top_plane_model * -1
            if np.dot(plane1_model,[0,0,1]) <0:
                plane1_model = plane1_model * -1


            ry = top_plane_model
            ry = ry / np.linalg.norm(ry)
            rx = p1 - p2
            rz = np.cross(rx,ry)
            
            rx =rx / np.linalg.norm(rx)
            rz =rz / np.linalg.norm(rz)
            vec1 = rz.copy()
            T = np.eye(4)
            T[:3, 0] = rx
            T[:3, 1] = ry
            T[:3, 2] = rz
            q = transformations.quaternion_from_matrix(T) 
            pose_radius = transformations.euler_from_matrix(T)
            pose_degree = np.rad2deg(pose_radius)

            rz = plane1_model
            rz = rz / np.linalg.norm(rz)
            vec2 = rz.copy()
            rx = p1 - p2
            ry = np.cross(rx,rz)

            rx =rx / np.linalg.norm(rx)
            ry =ry / np.linalg.norm(ry)
            
            
            T = np.eye(4)
            T[:3, 0] = rx
            T[:3, 1] = ry
            T[:3, 2] = rz
           # embed()
            #q = transformations.quaternion_from_matrix(T) 
            self.T = T
            pose_radius = transformations.euler_from_matrix(T)
            pose_degree_new = np.rad2deg(pose_radius)
            vec_angle = self.cali_angle(vec1,vec2)
            print("左上曲线")
            print(vec_angle)
            curve_p0_degree = pose_degree.copy()
            curve_p0_quaternion = q
            angle = self.cali_angle(plane1_model,top_plane_model)
            if angle > 90:
                angle = 180 -angle

            curve_p0_degree = angle.copy()
            # z_dir = plane1_model/ np.linalg.norm(plane1_model) *0.0655
            # self.left_top_curve_points_restored[p] = self.left_top_curve_points_restored[p] + z_dir
            curve_p0 = np.array([self.left_top_curve_points_restored[p][0],self.left_top_curve_points_restored[p][1],self.left_top_curve_points_restored[p][2],
                                vec1[0],vec1[1],vec1[2],vec2[0],
                                vec2[1],vec2[2] ])
            curve_p0_dis = np.array([self.left_top_curve_points_restored[p][0],self.left_top_curve_points_restored[p][1],self.left_top_curve_points_restored[p][2],
                                q[1],q[2], q[3],q[0] ])
            self.left_top_curve_path.append(curve_p0)
            self.left_top_curve_path_dis.append(curve_p0_dis)
        
        for p in self.left_top_curve_path_dis:
            self.v.Axes(p[:3],p[3:],0.002,3)

         # pass

    def find_left_top_line_path(self):

        #rz = (self.left_plane_model + self.top_plane_model_left)/2
        #rz = self.left_plane_model
        top_plane_model = self.top_plane_model.copy()
        left_plane_model = self.left_plane_model.copy()
        if np.dot(self.top_plane_model,[0,0,1]) > 0:
            top_plane_model = self.top_plane_model * -1
        if np.dot(self.left_plane_model,[0,0,1]) <0:
            left_plane_model = self.left_plane_model * -1


        #embed()
        ry = top_plane_model
        ry = ry / np.linalg.norm(ry)
        rx = self.max_point_on_left_top - self.min_point_on_left_top
        rz = np.cross(rx,ry)

        rx =rx / np.linalg.norm(rx)
        rz =rz / np.linalg.norm(rz)
        vec1 = rz.copy()
        
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ry
        T[:3, 2] = rz
        q = transformations.quaternion_from_matrix(T) 
        self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree = np.rad2deg(pose_radius)

        rz = left_plane_model
        rz = rz / np.linalg.norm(rz)
        vec2 = rz.copy()
        rx = self.max_point_on_left_top - self.min_point_on_left_top
        ry = np.cross(rx,rz)

        rx =rx / np.linalg.norm(rx)
        ry =ry / np.linalg.norm(ry)
        
        
        T = np.eye(4)
        T[:3, 0] = rx
        T[:3, 1] = ryrz =rz / np.linalg.norm(rz)
        T[:3, 2] = rz
      #  q = transformations.quaternion_from_matrix(T) 
        #self.T = T
        pose_radius = transformations.euler_from_matrix(T)
        pose_degree_new = np.rad2deg(pose_radius)

        vec_angle = self.cali_angle(vec1,vec2)
        print("左上线")
        print(vec_angle)
        self.left_top_line_degree = pose_degree.copy()
        self.left_top_line_quaternion = q
        angle = self.cali_angle(self.left_plane_model,self.top_plane_model)
        if angle > 90:
            angle = 180 -angle

        self.left_plane_top_angle = angle.copy()
        new_array_path = np.array([[vec1[0], vec1[1], vec1[2],
        vec2[0],vec2[1],vec2[2]]])

        # z_dir = left_plane_model/ np.linalg.norm(left_plane_model) *0.0655
        # self.left_top_line_points_restored = self.left_top_line_points_restored + z_dir
        self.left_top_line_path = np.concatenate((self.left_top_line_points_restored,
                         new_array_path.repeat(self.left_top_line_points_restored.shape[0], axis=0)), axis=1)
        new_array_dis = np.array([[q[1],q[2], q[3],q[0]]])
        self.left_top_line_path_dis =  np.concatenate((self.left_top_line_points_restored,
                         new_array_dis.repeat(self.left_top_line_points_restored.shape[0], axis=0)), axis=1)
        
        for p in self.left_top_line_path_dis:
            self.v.Axes(p[:3],p[3:],0.002,3)
        
        

    def find_segment_endpoints_left_top(self):
        # 初始化最大和最小距离及其对应的点
        top_dis = 0.05
        max_distance = -np.inf
        min_distance = np.inf
        self.max_point_on_left_top = None
        self.min_point_on_left_top = None
        for p in self.left_plane_top_inter_points_restored:
            # 计算点到前面平面的距离
            distance = np.abs(np.dot(self.front_plane_model_l,p - self.front_center_l))
            # 更新最大距离及其对应的点
            if distance > max_distance and distance < top_dis:
                max_distance = distance
                self.max_point_on_left_top = p
            
            # 更新最小距离及其对应的点
            if distance < min_distance:
                min_distance = distance
                self.min_point_on_left_top = p

    def find_segment_endpoints_left_bottom(self):
        # 初始化最大和最小距离及其对应的点
        bottom_dis = 0.1
        max_distance = -np.inf
        min_distance = np.inf
        self.max_point_on_left_bottom = None
        self.min_point_on_left_bottom = None
        for p in self.left_plane_bottom_inter_points_restored:
            # 计算点到前面平面的距离
            distance = np.abs(np.dot(self.front_plane_model_l,p - self.front_center_l))
            # 更新最大距离及其对应的点
            if distance > max_distance and distance < bottom_dis:
                max_distance = distance
                self.max_point_on_left_bottom = p
            
            # 更新最小距离及其对应的点
            if distance < min_distance:
                min_distance = distance
                self.min_point_on_left_bottom = p


    def find_segment_endpoints_front_top_l(self):
        # 初始化最大和最小距离及其对应的点

        max_distance = -np.inf
        min_distance = np.inf
        self.max_point_on_front_top_l = None
        self.min_point_on_front_top_l = None
        for p in self.front_plane_top_inter_points_l_restored:
            # 计算点到前面平面的距离
            distance = np.abs(np.dot(self.left_plane_model,p - self.left_center))
            # 更新最大距离及其对应的点
            if distance > max_distance :
                max_distance = distance
                self.min_point_on_front_top_l = p
            
            # 更新最小距离及其对应的点
            if distance < min_distance:
                min_distance = distance
                self.max_point_on_front_top_l = p

    def find_segment_endpoints_front_bottom_l(self):
        # 初始化最大和最小距离及其对应的点

        max_distance = -np.inf
        min_distance = np.inf
        self.max_point_on_front_bottom_l = None
        self.min_point_on_front_bottom_l = None
        for p in self.front_plane_bottom_inter_points_l_restored:
            # 计算点到前面平面的距离
            distance = np.abs(np.dot(self.left_plane_model,p - self.left_center))
            # 更新最大距离及其对应的点
            if distance > max_distance:
                max_distance = distance
                self.min_point_on_front_bottom_l = p
            
            # 更新最小距离及其对应的点
            if distance < min_distance:
                min_distance = distance
                self.max_point_on_front_bottom_l = p

    def find_segment_endpoints_surface_top_l(self):
        # 初始化最大和最小距离及其对应的点

        max_distance = np.inf
        min_distance = np.inf
        self.max_point_on_surface_top_l = None
        self.min_point_on_surface_top_l = None
        for p in self.surface_plane_top_inter_points_l_restored:
            # 计算点到前面平面的距离
            distance_min_left = np.linalg.norm(p - self.min_point_on_left_top)
            distance_max_front = np.linalg.norm(p - self.max_point_on_front_top_l)
            # 更新最大距离及其对应的点
            if distance_min_left < max_distance :
                max_distance = distance_min_left
                self.max_point_on_surface_top_l = p
            
            # 更新最小距离及其对应的点
            if distance_max_front < min_distance:
                min_distance = distance_max_front
                self.min_point_on_surface_top_l = p

    def find_segment_endpoints_surface_bottom_l(self):
        # 初始化最大和最小距离及其对应的点

        max_distance = np.inf
        min_distance = np.inf
        self.max_point_on_surface_bottom_l = None
        self.min_point_on_surface_bottom_l = None
        for p in self.surface_plane_bottom_inter_points_l_restored:
            # 计算点到前面平面的距离
            distance_min_left = np.linalg.norm(p - self.min_point_on_left_bottom)
            distance_max_front = np.linalg.norm(p - self.max_point_on_front_bottom_l)
            # 更新最大距离及其对应的点
            if distance_min_left < max_distance :
                max_distance = distance_min_left
                self.max_point_on_surface_bottom_l = p
            
            # 更新最小距离及其对应的点
            if distance_max_front < min_distance:
                min_distance = distance_max_front
                self.min_point_on_surface_bottom_l = p

    def build_left_plane_tree(self):
        self.left_plane_tree = KDTree(self.left_plane_points_in_left_coordinate)

    def build_left_front_tree(self):
        self.left_front_plane_tree = KDTree(self.front_plane_points_in_left_coordinate)
    
    def build_left_surface_tree(self):
        self.left_surface_plane_tree = KDTree(self.surface_plane_points_in_left_coordinate)

    def build_left_kdtree(self):
        
        self.build_left_plane_tree()
        self.build_left_surface_tree()
        self.build_left_front_tree()

    def transform_left_to_left_coordinate(self):
        self.transform_left_plane_to_left_coordinate()
        self.transform_left_front_to_left_coordinate()
        self.transform_left_surface_to_left_coordinate()

    def transform_left_plane_to_left_coordinate(self):
        self.left_plane_points_in_left_coordinate = (self.left_coordinate.T @ self.left_plane_points.T).T

    def transform_left_front_to_left_coordinate(self):
        self.front_plane_points_in_left_coordinate = (self.left_coordinate.T @ self.front_plane_points_l.T).T

    def transform_left_surface_to_left_coordinate(self):
        self.surface_plane_points_in_left_coordinate = (self.left_coordinate.T @ self.left_surface_points.T).T

    def find_left_weld_seam(self,tolerance = None, search_length = 0.03,top_dis = 0.08,bottom_dis = 0.1 ):
        self.find_three_points_from_left_plane()

        self.transform_left_to_left_coordinate()
        self.build_left_kdtree()

        self.find_point_group_in_left_plane()
        self.find_point_group_in_front_plane_l()
        self.find_point_group_in_surface_plane_l()

        self.restore_left_plane_point_group()
        self.transform_left_plane_point_group_to_left_coordinate()

        self.restore_front_plane_point_group_l()
        self.transform_front_plane_point_group_to_left_coordinate_l()

        self.restore_left_surface_plane_point_group()
        self.transform_left_surface_plane_point_group_to_left_coordinate()


        self.find_inter_points_from_left_plane_group()
        self.restore_left_plane_top_group()
        self.find_segment_endpoints_left_top()
        
        self.restore_left_plane_bottom_group()
        self.find_segment_endpoints_left_bottom()
        

        self.find_inter_points_from_front_plane_group_l()
        self.restore_front_plane_top_group_l()
        self.find_segment_endpoints_front_top_l()

        #self.find_segment_endpoints_left_front()
        self.restore_front_plane_bottom_group_l()
        self.find_segment_endpoints_front_bottom_l()

        self.find_inter_points_from_surface_plane_group_l()
        self.restore_surface_plane_top_group_l()
        self.find_segment_endpoints_surface_top_l()
        self.restore_surface_plane_bottom_group_l()
        self.find_segment_endpoints_surface_bottom_l()

        #找到了所有点，然后就找轨迹
        #查找左侧轨迹，分离3mm等分
        self.left_top_line_segmentation()
        self.transform_left_top_line_to_left_coordinate() 
        self.find_left_top_line_point_on_line()
        self.restore_left_top_line_points()
        self.restore_other_left_top_line_points()
        self.find_left_top_line_path()

        self.find_left_bottom_line_path()

        #截取上曲线，降采样
        self.left_top_curve_segmentation()
        self.transform_left_top_curve_to_left_coordinate() 
        self.find_left_top_curve_other_point_on_line()
        self.restore_left_top_curve_points()
        self.restore_other_left_top_curve_points()
        self.find_left_top_curve_path()

        self.find_left_bottom_curve_path()

     #   self.find_left_front_path()





        #定义起点和终点
      #  self.cut_left_welding_points(top_dis = 0.05,bottom_dis = 0.08)
        #self.down_sample_left
    def find_right_weld_seam(self,tolerance = None, search_length = 0.015 ):
        self.find_right_center_points(tolerance)
        self.find_right_welding_points(search_length)
        self.cut_right_welding_points(top_dis = 0.05,bottom_dis = 0.08)
        self.down_sample_right()


    def read_bottom_point_clouds(self,bottom_file_path = 'test13'):
        self.single_file_path = bottom_file_path
        self.bottom_point_clouds = self.read_single_point_cloud()
        return self.bottom_point_clouds
    
    def read_top_point_clouds(self,top_file_path = 'test15'):
        self.single_file_path = top_file_path
        self.top_point_clouds = self.read_single_point_cloud()
        return self.top_point_clouds

    def read_left_point_clouds(self,left_file_path='test8'):
        #读取点云
        self.single_file_path = left_file_path
        self.left_point_clouds = self.read_single_point_cloud()

        # self.left_point_clouds = Utils.UniformDownSample(pts=self.left_point_clouds,k=10)
        # self.left_point_clouds = Utils.VoxelDownSample(pts = self.left_point_clouds,voxel_size=0.003)
       # self.left_point_clouds,_ = Utils.StdOutlierRemoval(self.left_point_clouds,nb=100,ratio=2)
       # self.left_point_clouds,_ = Utils.RadiusOutlierRemoval(pts=self.left_point_clouds, nb_points=200, radius=0.005)
        
        return self.left_point_clouds

    def read_right_point_clouds(self,right_file_path='test19'):
        #读取点云，然后堆叠在一起
        self.single_file_path = right_file_path
        self.right_point_clouds = self.read_single_point_cloud()

        # self.right_point_clouds = Utils.UniformDownSample(pts=self.right_point_clouds,k=3)
        # self.right_point_clouds = Utils.VoxelDownSample(pts = self.right_point_clouds,voxel_size=0.001)
       # self.left_point_clouds,_ = Utils.StdOutlierRemoval(self.left_point_clouds,nb=10)
        
        return self.right_point_clouds


    def read_single_point_cloud(self):
        point_cloud = o3d.io.read_point_cloud(self.single_file_path +'/PointCloud.ply')
        pose = self.read_capture_pose(self.single_file_path +'/pose')
        target = self.transform_point_cloud(point_cloud,pose)
        pts = Utils.PCDToNumpy(target)
        pts = self.RemoveNaNs(pts)
        return pts

    def read_multi_point_cloud(self):
        merged_point_cloud = np.empty((0, 3), dtype=np.float32)
        for root,dirs,files in os.walk(self.multi_files_path):
            for dir_name in dirs:
                if dir_name.startswith("test"):
                    pose = self.read_capture_pose(os.path.join(root,dir_name,'pose'))
                    point_cloud = self.read_single_point_cloud(os.path.join(root,dir_name,'PointCloud.ply'))
                    target = self.transform_point(point_cloud,pose)
                    pts = Utils.PCDToNumpy(target)
                    merged_point_cloud = np.concatenate((merged_point_cloud, pts), axis=0)
        return  merged_point_cloud

    def project_point_to_line(self,point, line_point, line_direction):
        vec = point - line_point
        projection_length = np.dot(vec, line_direction)
        projection = line_point + projection_length * line_direction
        return projection

    def read_capture_pose(self,file_path):
        with open(file_path , 'r') as file:
            data = file.readline().split()
            pose = [float(num) for num in data]
        return pose
        
    def RemoveNaNs(self,pm):
        pm[np.where(np.isnan(pm[:,0]))] = [0,0,0]
        return pm

    def pose_to_mtx(self,pose):
        tmp = np.array([[1.  , 0.  , 0.  , 0.  ],
        [0.  , 1.  , 0.  , 0.  ],
        [0.  , 0.  , 1.  , 0.],
        [0.  , 0.  , 0.  , 1.  ]])
        tmp[:3,:3] = self.euler_to_rotation_matrix((pose[3],pose[4],pose[5]))
        tmp[:3,-1] = (pose[0],pose[1],pose[2])
        return tmp
    
    def euler_to_rotation_matrix(self,theta):
        """
        将欧拉角转换为旋转矩阵
        
        :param theta: 欧拉角列表，[rx, ry, rz]，单位为度数
        :return: 旋转矩阵，3x3的numpy数组
        """
        rx, ry, rz = np.radians(theta)
        
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
        
        R = Rz.dot(Ry.dot(Rx))
        
        return R

    def transform_point_cloud(self,point_cloud,pose):
        tcp2base = self.pose_to_mtx(pose)
        cam2base = np.matmul(tcp2base,self.hand_eye_cali)
        cam2base[:3,3:] = cam2base[:3,3:]/1000
        target = point_cloud.transform(cam2base)
        return target

    def find_plane(self,pts, threshold=0.0003):
        pts = pts[np.where(pts[:,2]!=0)]
        plane_model,inliers = Utils.FitPlane(pts, threshold=threshold)
        plane_points = pts[inliers]
        # 提取平面外的点
        non_plane_points = pts[~np.isin(np.arange(len(pts)), inliers)]
        plane_center = np.mean(plane_points,axis=0)
        return plane_model, plane_center ,plane_points,non_plane_points

    def project_points_to_plane(self,pts,plane_model,plane_center):

        rz = plane_model[:3]
        if plane_model[2] < 0:
            rz = -rz
        ry = np.cross(rz, [1, 0, 0])
        rx = np.cross(ry, rz)
        rx = rx / np.linalg.norm(rx)
        ry = ry / np.linalg.norm(ry)
        rz = rz / np.linalg.norm(rz)
        R = np.c_[rx, ry, rz]
        pts = (R.T @ pts.T).T  # 将点云垂直投影于拟合的平面
        plane_center = R.T @ plane_center

        return pts, plane_center, R


