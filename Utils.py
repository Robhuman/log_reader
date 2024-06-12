#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) RVBUST, Inc - All rights reserved.

import numpy as np
import cv2
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import math
import transformations
from loguru import logger as vision_log

def RemoveNaNs(pm):
    p = pm.reshape(-1, 3)
    return p[~np.isnan(p[:, 0])]


def GetROI(pm, roi):
    x, y, w, h = roi
    return pm[y:y+h, x:x+w]


def Transform(T, pts):
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def TransformationError(T_gt, T_est):
    R_gt = T_gt[:3, :3]
    R_est = T_est[:3, :3]
    rotation_error = abs(
        np.arccos(min(max(((np.matmul(R_gt.T, R_est)).trace() - 1) / 2, -1.0), 1.0)))

    t_gt = T_gt[:3, 3]
    t_est = T_est[:3, 3]
    translation_error = np.linalg.norm(t_gt - t_est)
    return rotation_error, translation_error


def UnitVector(vector):
    return vector / np.linalg.norm(vector)


def AngleBetween(v1, v2):
    v1_u = UnitVector(v1)
    v2_u = UnitVector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def NumpyToPCD(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def PCDToNumpy(pcd):
    return np.asarray(pcd.points)


def MatrixToRobotPose(T):
    position = T[:3, 3]
    R = np.eye(4)
    R[:3, :3] = T[:3, :3]
    q = transformations.quaternion_from_matrix(R)
    pos = [position[0], position[1], position[2],
           q[1], q[2], q[3], q[0]]
    return pos


def RobotPoseToMatrix(pose):
    T = np.eye(4)
    T[:3, 3] = [pose[0], pose[1], pose[2]]
    q = [pose[6], pose[3], pose[4], pose[5]]
    R = transformations.quaternion_matrix(q)
    T[:3, :3] = R[:3, :3]
    return T

def AdjustGamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0)**invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def ReadPointCloud(path, remove_nan=True):
    pcd = o3d.io.read_point_cloud(path, remove_nan_points=remove_nan)
    return PCDToNumpy(pcd)


def VoxelDownSample(pts, voxel_size=0.003, remove_nan=True):
    pts = RemoveNaNs(pts)
    pcd = NumpyToPCD(pts)
    p = pcd.voxel_down_sample(voxel_size=voxel_size)
    return PCDToNumpy(p)


def UniformDownSample(pts, k=3):
    p = NumpyToPCD(pts).uniform_down_sample(every_k_points=k)
    return PCDToNumpy(p)


def RandomDownSample(pts, k=3):
    pts_num = pts.shape[0]
    select_num = int(pts_num / k)
    select_idx = np.random.choice(pts_num, select_num)
    return pts[select_idx]


def ClusterDBSCAN(pc, eps=0.02, min_points=10):
    pcd = NumpyToPCD(pc)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    num_cluster = labels.max() + 1
    cluster_list = []
    for i in range(num_cluster):
        cluster_list.append(np.where(labels == i)[0])
    return cluster_list

def Truncate(pts, dim_indices, range):
    valid_ids = np.logical_and(pts[:, dim_indices] > range[0], pts[:, dim_indices] < range[1])
    return pts[valid_ids], valid_ids

def TruncateXYZ(pcd, xrange, yrange, zrange):
    total_range = np.arange(pcd.shape[0])
    valid_idx_x_ = np.logical_and(pcd[:, 0] > xrange[0], pcd[:, 0] < xrange[1])
    valid_idx_y_ = np.logical_and(pcd[:, 1] > yrange[0], pcd[:, 1] < yrange[1])
    valid_idx_z_ = np.logical_and(pcd[:, 2] > zrange[0], pcd[:, 2] < zrange[1])
    valid_idx_x = total_range[valid_idx_x_]
    valid_idx_y = total_range[valid_idx_y_]
    valid_idx_z = total_range[valid_idx_z_]
    valid_idx = np.intersect1d(
        valid_idx_z, np.intersect1d(valid_idx_x, valid_idx_y))
    return pcd[valid_idx], valid_idx


def GetBox(pcd):
    x_min = np.min(pcd[:, 0])
    x_max = np.max(pcd[:, 0])
    y_min = np.min(pcd[:, 1])
    y_max = np.max(pcd[:, 1])
    z_min = np.min(pcd[:, 2])
    z_max = np.max(pcd[:, 2])
    return x_min, x_max, y_min, y_max, z_min, z_max


def Point2PlaneDisSigned(point, plane_para):
    a, b, c, d = plane_para
    dis = (a*point[0]+b*point[1]+c*point[2]+d)/np.linalg.norm([a, b, c])
    return dis


def FitPlane(pc, threshold=0.001, iteration=1000):
    para, inlier = NumpyToPCD(pc).segment_plane(
        distance_threshold=threshold, ransac_n=3, num_iterations=iteration)
    return para, inlier

def TransformPlane(T, plane_para):
    norm = np.array(plane_para[:3]).reshape(-1, 3)
    z = -plane_para[-1] / plane_para[-2]
    norm_tran = Transform(np.linalg.inv(T).T, norm)
    norm_tran = UnitVector(norm_tran).flatten()
    point = np.array([0, 0, z]).reshape(-1, 3)
    point_tran = Transform(T, point).flatten()
    d = - norm_tran.dot(point_tran)
    para = [norm_tran[0], norm_tran[1], norm_tran[2], d]
    return para


def CheckPointCloudQuality(pc, depth_field, ratio_thres=0.5):
    error_msg = None
    pc = RemoveNaNs(pc)
    pc_center = np.mean(pc, axis=0)
    num_points = pc.shape[0]
    pc_valid, _ = Truncate(pc, 2, depth_field)
    valid_num = pc_valid.shape[0]
    if valid_num == 0:
        error_msg = "无效点云! 视野范围内没有点!!"
        vision_log.error(f"{error_msg}")
        if pc_center[2] < 0:
            error_msg = "相机位置过高! 请向下移动相机!"
            vision_log.error(f"{error_msg}")
        else:
            error_msg = "相机位置过低! 请向上移动相机!"
            vision_log.error(f"{error_msg}")
        return False, error_msg
    else:
        point_ratio = valid_num / num_points
        if point_ratio < ratio_thres:
            error_msg = f"无效点云! 视野范围内有效点比例过少({point_ratio*100:.2f}%)!! 请移动相机到合适位置!!"
            vision_log.error(f"{error_msg}")
            return False, error_msg
        else:
            vision_log.warning(f"valid point cloud:{point_ratio*100:.2f}%!")
            return True, error_msg

def AutoSingleCircleDetection(img_ori, min_gamma=0.8, max_gamma=1.5, threshold_block_size=50, threshold_c=10):
    from calib import detect_concentric_circles
    gamma_list = np.arange(min_gamma, max_gamma, 0.1)
    for gamma in gamma_list:
        vision_log.info(f"test gamma:{gamma}")
        img = AdjustGamma(img_ori, gamma)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        error_code, center_points_2d = detect_concentric_circles(
            img_grey, threshold_block_size, threshold_c)
        if error_code == 0:
            vision_log.info(f"success circle detection with gamma:{gamma}")
            break
    img_show = np.copy(img_ori)
    for center in center_points_2d:
        cv2.circle(img_show, tuple(np.int16(np.round(center))), 10,
                    (0, 0, 255), 10)
    return error_code, center_points_2d, img_show

def AutoCodeCircleDetection(img_ori, min_gamma=0.8, max_gamma=1.5):
    from calib import detect_calibration_pattern
    gamma_list = np.arange(min_gamma, max_gamma, 0.1)
    
    # coded pattern para
    board_width = 10
    board_height = 7
    calibration_pattern_type = 4
    is_separate_detect = False
    
    for gamma in gamma_list:
        vision_log.info(f"test gamma:{gamma}")
        img = AdjustGamma(img_ori, gamma)
        # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        error_code, center_points = detect_calibration_pattern(
            img,
            board_width=board_width,
            board_height=board_height,
            low_binary_threshold=30,
            high_binary_threshold=120,
            binarizing_images_flag=1,
            calibration_pattern_type=calibration_pattern_type,
            is_separate_detect=is_separate_detect)
        
        if error_code == 0:
            vision_log.info(f"success circle detection with gamma:{gamma}")
            break
    
    img_show = np.copy(img_ori)
    for k in np.arange(len(center_points)):
        cv2.circle(img_show, tuple(np.int16(np.round(center_points[k]))),
                    1, (255, 0, 0), 1)
        cv2.putText(img_show,
                    str(k),
                    tuple(np.int16(np.round(center_points[k]) + 10)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=3,
                    color=(0, 0, 255),
                    thickness=3)
    return error_code, center_points, img_show

def StdOutlierRemoval(pc, nb=20, ratio=2):
    pcd_std, idx = NumpyToPCD(pc).remove_statistical_outlier(
        nb_neighbors=nb, std_ratio=ratio)
    return PCDToNumpy(pcd_std), idx


def RadiusOutlierRemoval(pts, nb_points=20, radius=0.05):
    cl, ind = NumpyToPCD(pts).remove_radius_outlier(
        nb_points=nb_points, radius=radius)
    return PCDToNumpy(cl), ind


def WritePointCloud(path, pc, write_ascii=False, compressed=False):
    pcd = NumpyToPCD(pc)
    o3d.io.write_point_cloud(path, pcd, write_ascii=write_ascii, compressed=compressed)


def ComputeCurvature(pc):
    pc = pc[~np.isnan(pc)].reshape(-1, 3)
    T = np.eye(4)
    pca = PCA(n_components=3)
    pca.fit(pc)
    eigen_value = pca.singular_values_
    return eigen_value[2] / np.sum(eigen_value)


def PCATransformation(pc):
    pc = pc[~np.isnan(pc)].reshape(-1, 3)
    T = np.eye(4)
    pca = PCA(n_components=3)
    pca.fit(pc)
    R = pca.components_.T
    T[:3, :3] = R
    center = np.mean(pc, axis=0)
    T[:3, 3] = center

    # rotate if z_axis is different direction
    z_axis = [0, 0, 1]
    z_axis_cur = R[:, 2]
    angle = math.acos(
        np.dot(z_axis, z_axis_cur) / (np.linalg.norm(z_axis) * np.linalg.norm(z_axis_cur)))
    if angle > math.pi/2:
        T[:3, :3] = -T[:3, :3]

    return np.linalg.inv(T)


def NormalEstimation(point, pc, camera_pose=None, radius=0.01, surface=False):
    tree = KDTree(pc, leaf_size=2)
    idx = tree.query_radius(point.reshape(-1, 3), radius)[0].tolist()
    pc_nb = pc[idx]
    pca = PCA(n_components=3)
    pca.fit(pc_nb)
    normal = pca.components_[2, :].reshape(3, -1)
    
    if camera_pose is not None:
        # determin direction by camera pose
        point_direction = point - np.array(camera_pose)
        point_direction = point_direction / np.linalg.norm(point_direction)
        angle = math.acos(
            np.dot(point_direction, normal) / (np.linalg.norm(point_direction) * np.linalg.norm(normal)))
        if angle > math.pi/2:
            normal = -normal
    
    if surface and camera_pose is None:
        z_axis = np.array([0, 0, 1])
        angle = AngleBetween(z_axis, normal.flatten())
        if angle < math.pi/2:
            normal = -normal
        
    return normal


def Points2PlaneDist(pc, plane_para):
    return np.abs((np.array(plane_para[:3]).reshape(-1, 3) @ pc.T).T + plane_para[3])


def ProjectPoint2Plane(pc, plane_para):
    a, b, c, d = plane_para
    # a*0 + b*0 + c*z = d ---> z = -d / c
    z = -d / c
    point_in_plane = np.array([0, 0, z])
    normal = UnitVector(np.array(plane_para[:3])).reshape(3, -1)
    pc_direction = pc - point_in_plane
    dist = pc_direction @ normal
    pc_projected = pc - dist.reshape(-1, 1)*normal.reshape(1, -1)
        
    return pc_projected


def ExtractBoundary(pc, threshold=0.65, voxel_size=0.002):
    tree = KDTree(pc, leaf_size=2)
    idx = tree.query_radius(pc, 3*voxel_size)
    pc_boundary = []
    kp_idx = []
    for k in range(pc.shape[0]):
        pc_nb = pc[idx[k], :]
        if pc_nb.shape[0] < 3:
            continue
        pca = PCA(n_components=3)
        pca.fit(pc_nb)
        eigen_value = pca.singular_values_
        ratio_3 = eigen_value[1]/eigen_value[0]
        if ratio_3 < threshold:
            pc_boundary.append(pc[k, :])
            kp_idx.append(k)
    
    return np.array(pc_boundary).reshape(-1, 3), kp_idx


def ExtractCorner(pc, threshold=0.6, voxel_size=0.02):
    tree = KDTree(pc, leaf_size=2)
    idx = tree.query_radius(pc, 5*voxel_size)
    pc_boundary = []
    kp_idx = []
    for k in range(pc.shape[0]):
        pc_nb = pc[idx[k], :]
        if pc_nb.shape[0] < 3:
            continue
        pca = PCA(n_components=3)
        pca.fit(pc_nb)
        eigen_value = pca.singular_values_
        ratio_1 = eigen_value[2]/eigen_value[0]
        if ratio_1 > threshold:
            pc_boundary.append(pc[k, :])
            kp_idx.append(k)
    
    return np.array(pc_boundary).reshape(-1, 3), kp_idx


def ExtractEdge(pc, voxel_size=0.02, ratio3=0.8, ratio1_low=0.3, ratio1_hight=0.5):
    tree = KDTree(pc, leaf_size=2)
    idx = tree.query_radius(pc, 5*voxel_size)
    pc_boundary = []
    kp_idx = []
    for k in range(pc.shape[0]):
        pc_nb = pc[idx[k], :]
        if pc_nb.shape[0] < 3:
            continue
        pca = PCA(n_components=3)
        pca.fit(pc_nb)
        eigen_value = pca.singular_values_
        r1 = eigen_value[2]/eigen_value[0]
        r3 = eigen_value[1]/eigen_value[0]
        if r3 > ratio3 and r1 > ratio1_low and r1 < ratio1_hight:
            pc_boundary.append(pc[k, :])
            kp_idx.append(k)
    
    return np.array(pc_boundary).reshape(-1, 3), kp_idx

def ComputeScatterMatrix(point, pc_nb):
    scatter_matrix = np.zeros((3, 3))
    num = pc_nb.shape[0]
    for k in range(num):
        dif = pc_nb[k, :].reshape(3, -1) - point.reshape(3, -1)
        scatter_matrix = scatter_matrix + dif @ dif.T
    return scatter_matrix

def ExtractISSPoint(pc, salience_radius=0.02, non_max_radius=0.02, gamma_21=0.975, gamma_32=0.975, min_nb=20):
    tree = KDTree(pc, leaf_size=2)
    idx = tree.query_radius(pc, salience_radius)
    pc_boundary = []
    r3_all = []
    idx_interest = []
    for k in range(pc.shape[0]):
        pc_nb = pc[idx[k], :]
        if pc_nb.shape[0] < min_nb:
            r3_all.append(-100000)
            continue
        
        scatter_matrix = ComputeScatterMatrix(pc[k, :], pc_nb)
        eigen_value = np.linalg.eig(scatter_matrix)[0].tolist()
        eigen_value.sort(reverse=True)
        
        r21 = eigen_value[1]/eigen_value[0]
        r32 = eigen_value[2]/eigen_value[1]
        r3_all.append(eigen_value[2]) 
        if r21 < gamma_21 and r32 < gamma_32:
            pc_boundary.append(pc[k, :])
            idx_interest.append(k)
    
    idx = tree.query_radius(pc, non_max_radius)
    idx_final = []
    for k in idx_interest:
        is_max = True
        for j in idx[k]:
            if r3_all[k] < r3_all[j]:
                is_max = False
                break
        if is_max:
            idx_final.append(k)
    pc_boundary = pc[idx_final, :]
    
    return np.array(pc_boundary).reshape(-1, 3), idx_final

#############Registration###################
class CodeBoardDist:
    def __init__(self):
        self.A1 = 0.068
        self.A2 = 0.048
        self.A3 = 0.036
        self.A4 = 0.024
        self.A5 = 0.016
        self.A6 = 0.012


def GetCodeBoardCoordinate(board_width=10, board_height=7, cir_dis=CodeBoardDist().A4):
    num = board_height * board_width
    centers_3d = []
    for k in range(num):
        x = k % board_width
        y = int(k / board_width)
        centers_3d.append([x*cir_dis, y*cir_dis, 0])
    return centers_3d


def EstimateRigidTransformation3D(A, B):
    A = A.reshape(3, -1)
    B = B.reshape(3, -1)
    if A.shape != B.shape:
        vision_log.error(f"A and B should have the same size!!")
        return None

    num_rows, num_cols = A.shape
    if num_rows != 3:
        vision_log.error(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
        return None

    num_rows, num_cols = B.shape
    if num_rows != 3:
        vision_log.error(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")
        return None

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def ICPPoint2Point(pc_src, pc_dst, threshold=0.001, max_iteration=1000, trans_init=np.eye(4)):
    pcd_src = NumpyToPCD(pc_src)
    pcd_dst = NumpyToPCD(pc_dst)

    result = o3d.pipelines.registration.registration_icp(
        pcd_src, pcd_dst, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    return result.transformation, result.fitness


def ICPPoint2Plane(pc_src, pc_dst, threshold=0.001, max_iteration=1000, trans_init=np.eye(4), voxel_size=0.005):
    pcd_src = NumpyToPCD(pc_src)
    pcd_dst = NumpyToPCD(pc_dst)
    # estimate normals
    radius_normal = voxel_size * 2
    pcd_src.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_dst.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        pcd_src, pcd_dst, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    return result.transformation, result.fitness

def GeneralizeICP(pc_source, pc_target, threshold=0.001, max_iteration=1000, trans_init=np.eye(4)):
    result = o3d.pipelines.registration.registration_generalized_icp(
        NumpyToPCD(pc_source), NumpyToPCD(pc_target), threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    return result.transformation, result.fitness

def FGRRegistration(pcd_src, pcd_src_feature, 
                    pcd_dst, pcd_dst_feature, max_correspondence_dis=0.005):
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        pcd_src, pcd_dst, pcd_src_feature, pcd_dst_feature,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=max_correspondence_dis))
    return result.transformation

def computeFPFH(pc, voxel_size=0.001):
    pcd = NumpyToPCD(pc)

    # estimate normals
    radius_normal = voxel_size * 4
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # compute FPFH feature
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_fpfh

def RANSACRegistrationFeatureMatching(pcd_src, pcd_src_feature, 
                                      pcd_dst, pcd_dst_feature,
                                      max_correspondence_distance=0.005,
                                      max_iteration=10000,
                                      edge_length_thres=0.95,
                                      normal_angle_thes=math.pi/8,
                                      percentage=0.99):

    pcd_src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=20))
    pcd_dst.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=20))
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_src, pcd_dst, pcd_src_feature, pcd_dst_feature, True,
        max_correspondence_distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                edge_length_thres),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                max_correspondence_distance),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(normal_angle_thes)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, percentage))
    
    return result.transformation

def RANSACRegistrationCorrespondence(pcd_src, pcd_dst, corres,
                                      max_correspondence_distance=0.005,
                                      max_iteration=10000,
                                      edge_length_thres=0.95,
                                      normal_angle_thes=math.pi/8,
                                      percentage=0.99):
    pcd_src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=20))
    pcd_dst.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=20))
    corres = np.array(corres).reshape(-1, 2)
    corres_pcd = o3d.utility.Vector2iVector(corres)
    
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd_src, pcd_dst, corres_pcd, max_correspondence_distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                edge_length_thres),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                max_correspondence_distance),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(normal_angle_thes)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, percentage))
    
    # corres = np.array(corres).reshape(-1, 2)
    # corres_pcd = o3d.utility.Vector2iVector(corres)
    
    # result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
    #     pcd_src, pcd_dst, corres_pcd, max_correspondence_distance,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #     3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance)], 
    #     o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, percentage))

    return result.transformation

# ppf_config = {
#     "voxel_size_pre": 0.005,
#     "voxel_size": 0.005,
#     "uniform_downsample_interval": 3,
#     "m_rel_sample_dist": 0.03,
#     "m_calc_normal_relative": 0.05,
#     "m_invert_model_normal": False,
#     "m_rel_dense_sample_dist": 0.001,
#     "m_rel_dist_thresh": 0.005,
#     "m_angle_thresh_rad": 0.3,
#     "m_ref_mode_param": 1,
#     "m_fitness_thresh": 0.0,
#     "m_num_result": 1,
#     "m_voting_method": "using_all_points",
#     "m_edge_param": 100,
#     "m_edge_tolerance": 0.25,
#     "m_refine_method": "point2plane",
#     "m_refine_param_rel_dist_sparse_thresh": 3,
#     "m_refine_param_rel_dist_dense_thresh": 1,
#     "m_refine_param_dense_icp_iteration": 2000,
#     "ZAxis_Truncate_Threshold": [-100, 100],
#     "tcp_z_offset": 0}
    
def PPFRegistration(pc_src, pc_dst, config):
    import RVBUST.RVX.PoseEstimation as ppf
    # config
    ppf_config = ppf.PPF3DDetectorConfig()

    ppf_config.m_rel_sample_dist = config["m_rel_sample_dist"]
    ppf_config.m_calc_normal_relative = config["m_calc_normal_relative"]
    ppf_config.m_invert_model_normal = config["m_invert_model_normal"]
    ppf_config.m_rel_dense_sample_dist = config["m_rel_dense_sample_dist"]
    ppf_config.m_rel_dist_thresh = config["m_rel_dist_thresh"]
    ppf_config.m_angle_thresh_rad = config["m_angle_thresh_rad"]

    ppf_config.m_ref_mode.param = config["m_ref_mode_param"]
    ppf_config.m_fitness_thresh = config["m_fitness_thresh"]
    ppf_config.m_num_result = config["m_num_result"]
    if config["m_voting_method"].find("edge")!= -1:
        ppf_config.m_voting_mode.method = ppf.choose_voting_pts_method.using_edge_points
    if config["m_voting_method"].find("all") != -1:
        ppf_config.m_voting_mode.method = ppf.choose_voting_pts_method.using_all_sampled_points
    ppf_config.m_edge_param.pts_num = config["m_edge_param"]
    ppf_config.m_edge_param.tolerance=config["m_edge_tolerance"]

    ppf_config.m_refine_param.method = ppf.refine_method.point2plane
    ppf_config.m_refine_param.rel_dist_sparse_thresh = config[
        "m_refine_param_rel_dist_sparse_thresh"]
    ppf_config.m_refine_param.rel_dist_dense_thresh = config["m_refine_param_rel_dist_dense_thresh"]
    ppf_config.m_refine_param.dense_icp_iteration = config["m_refine_param_dense_icp_iteration"]
    ppf_detector = ppf.PPF3DDetector(ppf_config)

    # down sample
    pc_dst_down = VoxelDownSample(pc_dst, voxel_size=config["voxel_size"])

    # preprocessing data
    pc_dst_pre = ppf_detector.PreProcess(pc_dst_down, True)
    ppf_detector.Train(pc_dst_pre)
    pc_src_pre = ppf_detector.PreProcess(pc_src, False)

    # match
    results = ppf_detector.Match(pc_src_pre)
    if len(results) == 0:
        print("PPF doesn't get useful result!!!")
        return False, np.eye(4)

    for res in results:
        T = np.array(res.m_pose()).reshape(4, 4)

    return True, T

def TeaserRegistration(pc_src, pc_dst, noise_bound=0.05):
    import teaserpp_python
    pc_src = np.transpose(pc_src)
    pc_dst = np.transpose(pc_dst)

    # teaser parameter
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    # get result
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(pc_src, pc_dst)
    solution = solver.getSolution()
    T = np.eye(4)
    T[:3, :3] = solution.rotation
    T[:3, 3] = solution.translation

    return T

def BrustCorrespondeceEstimation(
    model_kp, model_kp_feature, sample_kp, sample_kp_feature):

    # best matching model -> sample
    sample_feature_tree = KDTree(sample_kp_feature)
    _, knn_idx = sample_feature_tree.query(model_kp_feature, k=1)
    BestMatchingModel2Sample = knn_idx[:, 0].tolist()

    # best matching sample -> model
    model_feature_tree = KDTree(model_kp_feature)
    _, knn_idx = model_feature_tree.query(sample_kp_feature, k=1)
    BestMatchingSample2Model = knn_idx[:, 0].tolist()

    # get correspondent point
    model_kp_cor = []
    sample_kp_cor = []
    for model_k in range(len(BestMatchingModel2Sample)):
        if model_k == BestMatchingSample2Model[BestMatchingModel2Sample[model_k]]:
            model_kp_cor.append(model_kp[model_k])
            sample_kp_cor.append(sample_kp[BestMatchingModel2Sample[model_k]])

    return np.array(model_kp_cor).reshape(-1, 3), np.array(sample_kp_cor).reshape(-1, 3)

def ProbReg(pc_source, pc_target, voxel_size=0.005):
    import probreg
    print("start probreg......")
    pc_source = VoxelDownSample(pc_source, voxel_size)
    pc_target = VoxelDownSample(pc_target, voxel_size)
    tf_para, _, _ = probreg.cpd.registration_cpd(NumpyToPCD(pc_source), NumpyToPCD(pc_target))
    # tf_para, _, _ = probreg.filterreg.registration_filterreg(NumpyToPCD(pc_source), NumpyToPCD(pc_target))
    T = np.eye(4)
    T[:3, :3] = tf_para.rot
    T[:3, 3] = tf_para.t
    
    return T


def RegistrationRANSAC(pc_source, pc_target, voxel_size=0.005, test_time=5, fit_res_mark=True):
    pc_source = VoxelDownSample(pc_source, voxel_size) 
    pc_target = VoxelDownSample(pc_target, voxel_size)
    
    feat_source = computeFPFH(pc_source, voxel_size) 
    feat_target = computeFPFH(pc_target, voxel_size)
    
    for k in range(test_time):
        pc_source_ = np.copy(pc_source)
        # ransac registration
        T = RANSACRegistrationFeatureMatching(NumpyToPCD(pc_source_), feat_source, 
                                        NumpyToPCD(pc_target), feat_target,
                                        max_correspondence_distance=1.5*voxel_size,
                                        max_iteration=20000, 
                                        edge_length_thres=0.95,
                                        percentage=0.7)
        pc_source_ = Transform(T, pc_source_)
        
        T_icp_coarse, res_fit = ICPPoint2Point(pc_source_, pc_target, threshold=4*voxel_size)
        T1 = T_icp_coarse @ T
        pc_source_ = Transform(T_icp_coarse, pc_source_)
        T_icp_refine, res_fit= ICPPoint2Point(pc_source_, pc_target, threshold=voxel_size)
        pc_source_ = Transform(T_icp_refine, pc_source_)
        T2 = T_icp_refine @ T1
        
        if fit_res_mark:
            if res_fit > 0.9:
                break
    
    # T_gicp = GeneralizeICP(pc_source, pc_target, 1.5*voxel_size, max_iteration=1000)
    # T2 = T_gicp @ T
    
    return T2

def RegistrationRANSACWithFeature(pc_source, depth_img_source, pc_target, depth_img_target, voxel_size=0.005):
    def FindPoints(pm, u, v):
        u = round(u)
        v = round(v)
        found_idx = [[u, v], [u, v-1], [u, v+1],
                    [u-1, v], [u-1, v-1], [u-1, v+1],
                    [u+1, v], [u+1, v-1], [u+1, v+1]]
        for idx in found_idx:
            point = pm[idx[1], idx[0]]
            if np.isnan(point[2]) == False:
                return point
        
        return None

    h, w = depth_img_source.shape[:2]
    pm_source = pc_source.reshape(h, w, -1)
    pm_target = pc_source.reshape(h, w, -1) 
    # kp1, kp2, matches = CannyDetection(depth_img_source, depth_img_target, match_ratio=0.7)
    # kp1, kp2, matches = SIFTDetection(depth_img_source, depth_img_target, match_ratio=0.8) ##size 10
    kp1, kp2, matches = ORBDetection(depth_img_source, depth_img_target, match_ratio=0.9) ## size 31
    pc_source_kp = []
    pc_target_kp = []
    for i, m in enumerate(matches):
        kp_source = kp1[m.queryIdx].pt
        kp_target = kp2[m.trainIdx].pt
        point_source = FindPoints(pm_source, kp_source[0], kp_source[1])
        point_target = FindPoints(pm_target, kp_target[0], kp_target[1])
        if (point_source is not None) and (point_target is not None):
            pc_source_kp.append(point_source)
            pc_target_kp.append(point_target)

    print(f"found useful pairs:{len(pc_source_kp)}/{len(matches)}")
    pc_source_kp = np.array(pc_source_kp).reshape(-1, 3)
    pc_target_kp = np.array(pc_target_kp).reshape(-1, 3)
    pc_source = RemoveNaNs(pc_source)
    pc_source = VoxelDownSample(pc_source, voxel_size)
    pc_target = RemoveNaNs(pc_target)
    pc_target = VoxelDownSample(pc_target, voxel_size)
    pc_source = np.r_[pc_source_kp, pc_source]
    pc_target = np.r_[pc_target_kp, pc_target]
    corres = [[i, i] for i in range(pc_source_kp.shape[0])]
    T = RANSACRegistrationCorrespondence(NumpyToPCD(pc_source), NumpyToPCD(pc_target), corres,
                                         max_correspondence_distance=3*voxel_size,
                                         edge_length_thres=0.8)
    
    pc_source = Transform(T, pc_source)
    
    T_icp_coarse, _ = ICPPoint2Point(pc_source, pc_target, threshold=4*voxel_size)
    T1 = T_icp_coarse @ T
    pc_source = Transform(T_icp_coarse, pc_source)
    T_icp_refine, _ = ICPPoint2Point(pc_source, pc_target, threshold=voxel_size)
    pc_source = Transform(T_icp_refine, pc_source)
    T2 = T_icp_refine @ T1
    
    # T_gicp = GeneralizeICP(pc_source, pc_target, 1.5*voxel_size, max_iteration=1000)
    # T2 = T_gicp @ T
    
    return T2


def RegistrationTeaser(pc_source, pc_target, voxel_size):
    pc_source = VoxelDownSample(pc_source, voxel_size) 
    pc_target = VoxelDownSample(pc_target, voxel_size)
    
    feat_source_pcd = computeFPFH(pc_source, voxel_size) 
    feat_target_pcd = computeFPFH(pc_target, voxel_size)
    
    feat_source = np.array(feat_source_pcd.data).reshape(feat_source_pcd.num(), -1)
    feat_target = np.array(feat_target_pcd.data).reshape(feat_target_pcd.num(), -1)
    
    pc_source_corr, pc_target_corr =  BrustCorrespondeceEstimation(pc_source, feat_source, pc_target, feat_target)
    print(f"get {pc_source_corr.shape[0]} correspondence")
    
    T = TeaserRegistration(pc_source_corr, pc_target_corr)
    
    return T


#########################opencv stuff#################################
def FlannMatching(des1, des2, idx_dict, search_dict):
    flann =  cv2.FlannBasedMatcher(idx_dict, search_dict)
    matches = flann.knnMatch(des1,des2,k=2)
    return matches

def ORBDetection(img_source, img_target, match_ratio=0.7):
    detector = cv2.ORB_create()
    kp1, des1 = detector.detectAndCompute(img_source, None)
    kp2, des2 = detector.detectAndCompute(img_target, None)
    FLANN_INDEX_LSH = 6
    index_dict= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    search_dict = dict(checks=50)
    # queryidx, trainidx
    matches_knn = FlannMatching(des1, des2, index_dict, search_dict)
    matches = []
    for i in range(len(matches_knn)):
        ms = matches_knn[i]
        first = ms[0]
        if len(ms) < 2:
            matches.append(first)
        else:
            second = ms[1]
            if first.distance<match_ratio*second.distance:
                matches.append(first)
    
    return kp1, kp2, matches
    
def SIFTDetection(img_source, img_target, match_ratio=0.7):
    detector = cv2.SIFT_create()
    kp1, des1 = detector.detectAndCompute(img_source, None)
    kp2, des2 = detector.detectAndCompute(img_target, None)
    FLANN_INDEX_KDTREE = 1
    index_dict = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_dict = dict(checks=50)
    matches_knn = FlannMatching(des1, des2, index_dict, search_dict)
    matches = []
    for i in range(len(matches_knn)):
        ms = matches_knn[i]
        first = ms[0]
        if len(ms) < 2:
            matches.append(first)
        else:
            second = ms[1]
            if first.distance<match_ratio*second.distance:
                matches.append(first)
    return kp1, kp2, matches

def Canny(img, low_thres, high_thres, nb_size=1):
    img_edg = cv2.Canny(img.astype(np.uint8), low_thres, high_thres)
    kp_idx = np.where(img_edg>0)
    kp_num = kp_idx[0].shape[0]
    kp = []
    for k in range(kp_num):
        kp.append(cv2.KeyPoint(float(kp_idx[1][k]), float(kp_idx[0][k]), nb_size))
    return kp

def CannyDetection(depth_img_source, depth_img_target, match_ratio=0.7):
    kp1 = Canny(depth_img_source, 10, 20)
    kp2 = Canny(depth_img_target, 10, 20)
    detector = cv2.SIFT_create()
    des1 = detector.compute(depth_img_source, kp1)
    des2 = detector.compute(depth_img_target, kp2)
    FLANN_INDEX_KDTREE = 1
    index_dict = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_dict = dict(checks=50)
    matches_knn = FlannMatching(des1[1], des2[1], index_dict, search_dict)
    matches = []
    for i in range(len(matches_knn)):
        ms = matches_knn[i]
        first = ms[0]
        if len(ms) < 2:
            matches.append(first)
        else:
            second = ms[1]
            if first.distance<match_ratio*second.distance:
                matches.append(first)
    return kp1, kp2, matches

class BlobDetectionParams():
    def __init__(self):
        # Change thresholds
        self.minThreshold = 0
        self.maxThreshold = 255

        # Filter by color
        self.filterByColor = True
        self.blobColor = 255
        
        # Filter by Area.
        self.filterByArea = True
        self.minArea = 100
        self.maxArea = 1000000

        # Filter by Circularity
        self.filterByCircularity = True
        self.minCircularity = 0.5
        self.maxCircularity = 1

        # Filter by Convexity
        self.filterByConvexity = True
        self.minConvexity = 0.1
        self.maxConvexity = 1

        # Filter by Inertia
        self.filterByInertia = True
        self.minInertiaRatio = 0.01
        self.maxInertiaRatio = 1

def BlobDetection(img, blob_para=BlobDetectionParams()):
    img = cv2.GaussianBlur(img, (5, 5), 1, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = blob_para.minThreshold #0
    params.maxThreshold = blob_para.maxThreshold #255

    # Filter by color
    params.filterByColor = blob_para.filterByColor #True
    params.blobColor = blob_para.blobColor #255
    
    # Filter by Area.
    params.filterByArea = blob_para.filterByArea #True
    params.minArea = blob_para.minArea #1000
    params.maxArea = blob_para.maxArea #1000000

    # Filter by Circularity
    params.filterByCircularity = blob_para.filterByCircularity #True
    params.minCircularity = blob_para.minCircularity #0.5
    params.maxCircularity = blob_para.maxCircularity #1

    # Filter by Convexity
    params.filterByConvexity = blob_para.filterByConvexity #True
    params.minConvexity = blob_para.minConvexity #0.1
    params.maxConvexity = blob_para.maxConvexity #1

    # Filter by Inertia
    params.filterByInertia = blob_para.filterByInertia #True
    params.minInertiaRatio = blob_para.minInertiaRatio #0.01
    params.maxInertiaRatio = blob_para.maxInertiaRatio #1
    
    detector = cv2.SimpleBlobDetector_create(params)
    
    keypoints = detector.detect(img)
    
    img_show = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    center_list = []
    for k in range(len(keypoints)):
        x, y = keypoints[k].pt
        x = round(x)
        y = round(y)
        r = round( keypoints[k].size / 2)
        cv2.circle(img_show, (x, y), r, (0, 255, 0), 4)
        center_list.append([x, y])
    # center_list = sorted(center_list, key=lambda k:[k[0], k[1]])
    
    return img_show, center_list

################path relative#####################
def TransformPaths(T, paths):
    new_paths = []
    for path in paths:
        new_path = []
        for point in path:
            T_cur = np.eye(4)
            T_cur[0:3, 3] = [point[0], point[1], point[2]]
            q_cur = [point[6], point[3], point[4], point[5]]
            R = transformations.quaternion_matrix(q_cur)
            T_cur[0:3, 0:3] = R[0:3, 0:3]
            T_final = T @ T_cur

            new_R = np.eye(4)
            new_R[:3, :3] = T_final[:3, :3]
            new_q = transformations.quaternion_from_matrix(new_R)
            new_point = [T_final[0, 3], T_final[1, 3], T_final[2, 3], 
                        new_q[1], new_q[2], new_q[3], new_q[0]]
            new_path.append(new_point)
        new_paths.append(new_path)
    return new_paths

def PathGeneration(points, pc, camera_pose, radius=0.01, surface=False, z_direction=None):
    point_num = len(points)
    path = []
    for k in range(point_num):
        if z_direction is None:
            z_direction = NormalEstimation(np.array(points[k]), pc, camera_pose, radius, surface)
        z_direction = UnitVector(z_direction).reshape(1, -1)
        if k == (point_num-1):
            move_dir = np.array(points[k]) - np.array(points[k-1])
        else:
            move_dir = np.array(points[k+1]) - np.array(points[k])
        y_direction = np.cross(z_direction, move_dir.reshape(1, -1))
        x_direction = np.cross(y_direction, z_direction)
        y_direction = UnitVector(y_direction)
        x_direction = UnitVector(x_direction)
        T = np.eye(4)
        T[:3, 0] = x_direction
        T[:3, 1] = y_direction
        T[:3, 2] = z_direction
        q = transformations.quaternion_from_matrix(T) 
        pos = [points[k][0], points[k][1], points[k][2],
               q[1], q[2], q[3], q[0]]
        path.append(pos)
    return path

def GetManualPath(pc, viewer, camera_pose, radius=0.01, surface=False):
    paths = []
    from RVBUST import Vis
    viewer.SetIntersectorMode(Vis.IntersectorMode_Point)
    while True:
        select_points = []
        q = input(f"press enter to generate path(enter q to quit)...")
        if q.lower() == "q":
            break
        mid_point_tmp = viewer.GetPickedPointAxes()
        for p in mid_point_tmp:
            select_points.append(viewer.GetPosition(p)[1])
        viewer.ClearPickedPointAxes()
        
        path = PathGeneration(select_points, pc, camera_pose, radius, surface)
        paths.append(path)
    viewer.SetIntersectorMode(Vis.IntersectorMode_Disable)
    return paths

def VisualizePath(paths, viewer, scale=0.1):
    h_list = []
    for path in paths:
        for pose in path:
            position = pose[:3]
            q = [pose[6], pose[3], pose[4], pose[5]]
            R = transformations.quaternion_matrix(q)[:3, :3]
            color = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
            for k in range(3):
                axis_point = np.array(position).reshape(1, -1) + scale * R[:3, k].reshape(1, -1)
                axis_point = axis_point.tolist()[0]
                h_list.append(viewer.Line([position[0], position[1], position[2], 
                                           axis_point[0], axis_point[1], axis_point[2]], 2, color[k]))
    return h_list

def VisualizeCoordinate(viewer):
    handle = []
    handle.append(viewer.Line([0, 0, 0, 1, 0, 0], 3, (1, 0, 0)))
    handle.append(viewer.Line([0, 0, 0, 0, 1, 0], 3, (0, 1, 0)))
    handle.append(viewer.Line([0, 0, 0, 0, 0, 1], 3, (0, 0, 1)))
    return handle

def VisualizePose(T, viewer, scale=0.1, line_size=2):
    R = T[:3, :3]
    position = T[:3, 3]
    color = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
    h_list = []
    for k in range(3):
        axis_point = np.array(position).reshape(1, -1) + scale * R[:3, k].reshape(1, -1)
        axis_point = axis_point.tolist()[0]
        h_list.append(viewer.Line([position[0], position[1], position[2], 
                                    axis_point[0], axis_point[1], axis_point[2]], line_size, color[k]))
    return h_list
    
def GetDepthMap(pc, T, img, binary=True, trunc=[0.01, 10]):
    # rotate to cut plane
    pc = Transform(T, pc)
    pc_trun, idx = Truncate(pc, 2, trunc)
    
    pc = Transform(np.linalg.inv(T), pc)
    pc_trun = Transform(np.linalg.inv(T), pc_trun)
    in_idx = np.arange(pc.shape[0])[idx].tolist()
    out_idx = np.arange(pc.shape[0])[~idx].tolist()
    
    if binary:
        pc_ori = np.copy(pc)
        pc_ori[out_idx] = math.nan
        pc[out_idx, 2] = 0
        pc[in_idx, 2] = 255
        h, w = img.shape[:2]
        pm = pc.reshape(h, w, -1)
        
        depth_normal = pm[:, :, 2]
        return pc_ori, depth_normal
        
    pc_trun = RemoveNaNs(pc_trun)
    pc_ori = np.copy(pc)
    pc_ori[out_idx] = math.nan
    min_z = np.min(pc_trun, axis=0)[2]
    max_z = np.max(pc_trun, axis=0)[2]
    pc[out_idx, 2] = min_z
    h, w = img.shape[:2]
    pm = pc.reshape(h, w, -1)
    depth = pm[:, :, 2]
    depth_normal = 255 * (depth - min_z) / (max_z - min_z)
    
    return pc_ori, depth_normal