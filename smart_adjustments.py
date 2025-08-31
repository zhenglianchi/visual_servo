import numpy as np
def generate_target_points_auto_scale(current_corners, current_tvec, target_distance, image_resolution):
    """
    根据当前角点和距离，生成在 target_distance 下的期望角点（免标定）

    参数：
    - current_corners: (4, 2) 当前检测到的四个角点
    - current_tvec: (3,) 当前 tvec，用于计算当前距离
    - target_distance: float，期望距离（米）
    - image_resolution: (width, height)

    返回：
    - target_points: (4, 2) 期望的四个角点（居中，按目标距离缩放）
    """
    current_distance = np.linalg.norm(current_tvec)
    # 当前包围框的宽度和高度
    x_coords = current_corners[:, 0]
    y_coords = current_corners[:, 1]
    current_width = max(x_coords) - min(x_coords)
    current_height = max(y_coords) - min(y_coords)

    # 计算在 target_distance 下应有的尺寸
    scale = current_distance / target_distance  # 距离越远，图像越小；越近越大
    #print("current_distance: ",current_distance)
    #print("target_distance: ",target_distance)
    #print("scale: ",scale)
    expected_width = current_width * scale
    expected_height = current_height * scale

    # 图像中心
    center_x = image_resolution[0] // 2
    center_y = image_resolution[1] // 2

    half_w = int(expected_width / 2)
    half_h = int(expected_height / 2)

    target_points = np.array([
        [center_x - half_w, center_y - half_h],  # 左上
        [center_x + half_w, center_y - half_h],  # 右上
        [center_x + half_w, center_y + half_h],  # 右下
        [center_x - half_w, center_y + half_h]   # 左下
    ])

    return target_points