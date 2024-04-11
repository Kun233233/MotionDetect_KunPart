import numpy as np
import cv2 as cv

import sys
import time
sys.path.append("D:/conda_envs/main-py38/Lib/site-packages/boost_python_cam")
import BoostPythonCam


def GetPixels(file_path):
    # 读取文本文件
    with open(file_path, 'r') as f:
        # 逐行读取文件内容
        lines = f.readlines()

    # 初始化空列表用于存储坐标数据
    coordinates = []

    # 遍历每一行
    for line in lines[1:]:
        # 初始化当前行坐标列表
        current_row = []
        # 移除行末的换行符并按制表符分割字符串
        parts = line.strip().split('\t')
        
        # 如果当前行为空，则手动设置默认值
        if not parts[1:]:
            for i in range(68): # TODO(kun): 20240410 暂时写死
                # 默认设置[-1, -1]
                current_row.append([-1, -1])
        else:
            # 遍历除了第一个元素（PicNumber）之外的所有元素
            for part in parts[1:]:
                # 移除双引号，并按逗号分割字符串
                xy = part.replace('"', '').replace('(', '').replace(')', '').split(',')
                # 将坐标转换为整数并添加到坐标列表中
                current_row.append([int(xy[0]), int(xy[1])])

        # 将当前行添加到保存所有帧的列表中
        coordinates.append(current_row)

    # 将坐标列表转换为 NumPy 数组
    coordinates_array = np.array(coordinates)

    return coordinates_array

def GetCountMax(file_path):
    try:
        with open(file_path, 'r') as rgb_count_file:
            # 读取文件中的数字
            countmax = int(rgb_count_file.read())
    except FileNotFoundError:
        print("Unable to open the file.")
        exit(1)
    return countmax

def GetImgsIndex(countmax):
    # 生成图片文件序号
    file_index = []
    for i in range(countmax):
        name = str(i)
        t = len(name)  # 存储当前序号的数字位数
        
        for j in range(6 - t):
            name = "0" + name
        file_index.append(name)
        # print(name)
    return file_index



# image = cv.imread("C:/Users/Kun/Desktop/1.jpg", cv.IMREAD_GRAYSCALE)
    # b, dst = cv.threshold(image, 100, 255, cv.THRESH_BINARY)
# rgb_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/RGBImgs1/rgb_000120.png"
# depth_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/DepthImgs1/depth_000120.png"


if __name__ == '__main__':


    camera_param_file_path = "D:/aaaLab/aaagraduate/SaveVideo/source/CameraParams.json"

    # feature_pixels_position = pixels_position[:, [31-1, 37-1, 40-1, 43-1, 46-1, 52-1], :]


    count = 0 # 记录循环轮次
    # feature_points = [] # 记录每一次循环特征点对应三维空间坐标
    # pose = [] # 记录每一帧头部位姿
    
    # 初始化RGBCamera
    cam_tool = BoostPythonCam.RGBDCamera(camera_param_file_path, True, 0)

    while(1):
        start_time = time.time()
        cam_tool.get_img_from_cam()

        cam_tool.registration_capturedimg()


        depth_inrgb_CV8U = cam_tool.get_mat("depth_inrgb_CV8U")
        rgb_depth = cam_tool.get_mat("rgb_depth")
        rgb = cam_tool.get_mat("rgb")
        # depth_inrgb_CV8U = cam_tool.get_mat("depth_inrgb_CV8U")

        cv.imshow("depth_inrgb_CV8U", depth_inrgb_CV8U)
        cv.imshow("rgb_depth", rgb_depth)
        cv.imshow("rgb", rgb)

        cv.waitKey(1)
        count += 1

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
