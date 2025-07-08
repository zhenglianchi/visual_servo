import numpy as np
import cv2
import time
import pyrealsense2 as rs
from Servo import servo
from UR_Base import UR_BASE
import cv2
import cv2.aruco as aruco

def resize_and_center_box(target_points, image_size, padding=0):
    if len(target_points) != 4:
        raise ValueError("目标框必须包含四个点。")

    points = np.array(target_points, dtype=np.float32)

    center = np.mean(points, axis=0)
    image_center = np.array([image_size[0] / 2, image_size[1] / 2])

    moved_points = points + (image_center - center)

    v1 = moved_points[1] - moved_points[0]
    v2 = moved_points[3] - moved_points[0]

    width = np.linalg.norm(v1)
    height = np.linalg.norm(v2)

    half_w = width / 2
    half_h = height / 2

    rect_points = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h]
    ])

    rect_points += image_center

    center_rect = np.mean(rect_points, axis=0)
    rect_points = np.array([
        [
            rect_points[i][0] + (rect_points[i][0] - center_rect[0]) * padding / max(width, height),
            rect_points[i][1] + (rect_points[i][1] - center_rect[1]) * padding / max(width, height)
        ]
        for i in range(4)
    ])

    x_coords = [p[0] for p in rect_points]
    y_coords = [p[1] for p in rect_points]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    rect_points = [
        [x_min, y_min],  # 左上
        [x_max, y_min],  # 右上
        [x_max, y_max],  # 右下
        [x_min, y_max]   # 左下
    ]

    return rect_points

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

    HOST = '192.168.111.10'

    # 初始化UR5机械位姿
    first_tcp = np.array([-0.500, -0.309, 0.723, 1.052, 2.361, -2.665])

    ur5 = UR_BASE(HOST,first_tcp)

    time.sleep(5)

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

            target_points = resize_and_center_box(detected_points,resolution)

            for point in target_points:
                cv2.circle(img_color, point, 3, (255, 255, 255), -1)

            uv = np.array(detected_points).T
            p_star = np.array(target_points).T

            print("检测点为:\n",uv)
            print("目标点为:\n",p_star)
            
            #try:
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




