import numpy as np
import cv2
import time
import pyrealsense2 as rs
from Servo import servo
from UR_Base import UR_BASE
import cv2
import cv2.aruco as aruco
from smart_adjustments import generate_target_points_auto_scale

def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame()
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics
    img_color = np.asanyarray(aligned_color_frame.get_data())
    img_depth = np.asanyarray(aligned_depth_frame.get_data())
    intr_matrix = np.array([
        [color_intrin.fx, 0, color_intrin.ppx], [0, color_intrin.fy, color_intrin.ppy], [0, 0, 1]
    ])
    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame, np.array(color_intrin.coeffs), intr_matrix


if __name__ == "__main__":
    global img_color
    target_distance = 0.45  # 米（你想停在多远，就改这里）

    HOST = '192.168.111.10'

    # 初始化UR5机械位姿
    first_tcp = np.array([0.029125568130258156, -0.21169816738266292, 0.8085129143669209, -0.7182055727080571, 1.9599717985725258, -1.6564796217293596])

    ur5 = UR_BASE(HOST, fisrt_tcp=first_tcp)

    time.sleep(20)
    # 控制增益
    lambda_gain = np.array([0.008, 0.008, 0.008, 0.007, 0.007, 0.007])

    # 构造对角矩阵
    lambda_gain = np.diag(lambda_gain)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # 定义全局变量，存储鼠标点击位置的坐标和半径
    target_point = None

    start_time = time.time()  # 获取程序开始时间

    detected_points = None

    target_points = None

    center_point = None

    cv2.namedWindow("RealSence")

    while True:
        color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame, intr_coeffs, intr_matrix = get_aligned_images()

        f = [color_intrin.fx,color_intrin.fy]
        resolution = [color_intrin.width,color_intrin.height]

        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        # 创建detector parameters
        parameters = aruco.DetectorParameters_create()
        # 输入rgb图, aruco的dictionary, 相机内参, 相机的畸变参数
        corners, ids, rejected_img_points = aruco.detectMarkers(img_color, aruco_dict, parameters=parameters,cameraMatrix=intr_matrix, distCoeff=intr_coeffs)
        # rvec是旋转向量， tvec是平移向量
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.094, intr_matrix, intr_coeffs)
        if rvec is not None and tvec is not None:
            aruco.drawDetectedMarkers(img_color, corners)
            detected_points = corners[0][0]

            average_x = (detected_points[0][0] + detected_points[1][0] + detected_points[2][0] + detected_points[3][0]) / 4
            average_y = (detected_points[0][1] + detected_points[1][1] + detected_points[2][1] + detected_points[3][1]) / 4
            # 得到中心点坐标
            center_point = (average_x, average_y)

            # ✅ 使用免标定方法生成目标点
            target_points = generate_target_points_auto_scale(
                current_corners=corners[0][0],
                current_tvec=tvec[0][0],
                target_distance=target_distance,
                image_resolution=resolution
            )

            for point in target_points:
                cv2.circle(img_color, point, 3, (255, 255, 255), -1)

            uv = np.array(detected_points).T
            p_star = np.array(target_points).T

            #print("检测点为:\n",uv)
            #print("目标点为:\n",p_star)
            
            servo(ur5,uv,img_depth,p_star,lambda_gain,f,resolution,center_point)
        else:
            detected_points = None
            
        cv2.imshow('RealSence', img_color)
        key = cv2.waitKey(1)
        if key == 27 :
            ur5.disconnect()
            pipeline.stop()
            cv2.destroyAllWindows()
            break




