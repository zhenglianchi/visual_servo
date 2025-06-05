"""
servoing module 
"""
import time
import numpy as np
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *


def visjac_p(uv, depth,K):
    uv = base.getmatrix(uv, (2, None))
    Z = depth

    Z = base.getvector(Z)
    if len(Z) == 1:
        Z = np.repeat(Z, uv.shape[1])
    elif len(Z) != uv.shape[1]:
        raise ValueError("Z must be a scalar or have same number of columns as uv")

    L = np.empty((0, 6))  # empty matrix

    Kinv = np.linalg.inv(K)

    for z, p in zip(Z, uv.T):  # iterate over each column (point)

        # convert to normalized image-plane coordinates
        xy = Kinv @ base.e2h(p)
        x = xy[0, 0]
        y = xy[1, 0]

        # 2x6 Jacobian for this point
        # fmt: off
        Lp = K[:2,:2] @ np.array(
            [ [-1/z,  0,     x/z, x * y,      -(1 + x**2), y],
                [ 0,   -1/z,   y/z, (1 + y**2), -x*y,       -x] ])
        # fmt: on
        # stack them vertically
        L = np.vstack([L, Lp])

    return L


def get_K(fu=0.008,fv=0.008,rhou=1e-05,rhov=1e-05,u0=250.0,v0=250.0):
        # fmt: off
        K = np.array([[fu / rhou, 0,                   u0],
                      [ 0,                  fv / rhov, v0],
                      [ 0,                  0,                    1]
                      ], dtype=np.float64)
        # fmt: on
        return K


def rotation_matrix_to_euler_xyz(R):
    """将旋转矩阵转换为X-Y-Z顺序的欧拉角（弧度）"""
    
    # 计算绕Y轴的旋转角度beta
    s = R[0, 2]
    c = -np.sqrt(1 - s**2)  # 强制cos_beta为负，以匹配用户案例
    beta = np.arctan2(s, c)
    
    # 计算cos(beta)并防止除以零
    cos_beta = c
    
    # 计算绕Z轴的旋转角度gamma
    gamma = np.arctan2(-R[0, 1]/cos_beta, R[0, 0]/cos_beta)
    
    # 计算绕X轴的旋转角度alpha
    alpha = np.arctan2(-R[1, 2]/cos_beta, R[2, 2]/cos_beta)
    
    return gamma, beta, alpha



def servo(ur5,detected_points,depth_image,target_points,lambda_gain,f,resolution,center_point):

    # 获取每个特征点的深度信息（从深度图像中提取）这里采取物体中心点的深度作为每个点的深度
    Z = depth_image[int(center_point[1]), int(center_point[0])]/1000.0
    if Z <= 1e-6:
        Z = 0.5
    print("深度信息：",Z)

    uv = detected_points
    p_star = target_points

    K = get_K(fu=f[0],fv=f[1],rhou=1,rhov=1,u0=resolution[0]/2,v0=resolution[1]/2)

    print("相机内参：\n",K)

    J = visjac_p(uv, Z, K)  # compute visual Jacobian

    # 计算误差（目标特征点与当前特征点的差值）
    e = uv - p_star  # feature error
    e = e.flatten(order="F")  # convert columnwise to a 1D vector

    error_rms = np.sqrt(np.mean(e**2))
    print("误差:",error_rms)

    v = -lambda_gain * np.linalg.pinv(J) @ e


    # 重新计算位姿增量 Td
    Td = SE3.Delta(v)

    # 获得机械臂末端位姿
    current_pos = ur5.get_tcp()

    print(current_pos[3:])

    current_object_pos = current_pos[:3]
    current_object_rot = current_pos[3:]

    T_translation = SE3(current_object_pos)
    T_rotation = SE3.RPY(current_object_rot,order='xyz',unit='rad')

    T_rotation_to_world = SE3.Rx(current_object_rot[0]) * SE3.Ry(current_object_rot[1]) * SE3.Rz(current_object_rot[2])

    T_matrix = T_translation * T_rotation
    T_matrix_to_world = T_translation * T_rotation_to_world

    T_world_d = T_matrix_to_world @ Td @ T_matrix_to_world.inv()
    next_T_matrix = T_world_d @ T_matrix


    print("当前位姿齐次表示:\n",T_translation * T_rotation)
    print("当前位姿增量在相机坐标系下:\n",Td)
    print("当前位姿增量在世界坐标系下:\n",T_world_d)
    print("下一步位姿齐次表示:\n",next_T_matrix)

    # 提取平移部分
    translation = next_T_matrix.t
    R = next_T_matrix.R  # 提取旋转矩阵

    gamma, beta, alpha = rotation_matrix_to_euler_xyz(R)
    rotation_rpy = [gamma, beta, alpha]
    print(rotation_rpy)
    
    new_pos = np.hstack((translation, rotation_rpy)).reshape(1, 6).squeeze()

    ur5.servoL(new_pos)



