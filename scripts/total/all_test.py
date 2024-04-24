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

    # # ------------------- 20240326 1 ------------------- 
    depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/DepthImgs1"
    rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/RGBImgs1"
    feature_rgb_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/1_68.txt"
    feature_3D_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/points1.txt"
    motion_vec_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/motion1.txt"
    # # ------------------- end ------------------- 

    # ------------------- 20240326 1 ------------------- 
    # depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/DepthImgs3"
    # rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/RGBImgs3"
    # feature_rgb_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/3_68.txt"
    # feature_3D_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/points3.txt"
    # motion_vec_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/motion3.txt"
    # ------------------- end ------------------- 
    camera_param_file_path = "D:/aaaLab/aaagraduate/SaveVideo/source/CameraParams.json"


    pixels_position = GetPixels(feature_rgb_path)
    # feature_pixels_position = np.concatenate((pixels_position[:,31-1,:], pixels_position[:,37-1,:], 
    #                                           pixels_position[:,40-1,:], pixels_position[:,43-1,:], 
    #                                           pixels_position[:,46-1,:], pixels_position[:,52-1,:]), 
    #                                           axis=1)
    # feature_pixels_position = pixels_position[:, [31-1, 37-1, 40-1, 43-1, 46-1, 52-1], :]
    # feature_pixels_position = pixels_position[:, [31, 37, 40, 43, 46, 52], :]

    feature_pixels_position = pixels_position[:, [22-1, 23-1, 30-1, 34-1, 37-1, 40-1, 43-1, 46-1, 49-1, 52-1, 55-1, 58-1], :]
    
    # print(pixels_position.shape)
    # print(feature_pixels_position.shape)


    frame_path = rgb_folder_path + "/frame_num.txt"
    countmax = GetCountMax(frame_path) # 读取保存图片的数目，即帧数
    imgs_index = GetImgsIndex(countmax) # 生成对应帧的图片序列名
    # countmax = 3

    count = 0 # 记录循环轮次
    feature_points = [] # 记录每一次循环特征点对应三维空间坐标
    pose = [] # 记录每一帧头部位姿
    
    # 初始化RGBCamera
    cam_tool = BoostPythonCam.RGBDCamera(camera_param_file_path, False, 0)

    for i in range(countmax):
        start_time = time.time()
        rgb_file_name = rgb_folder_path + "/rgb_" + imgs_index[i] + ".png"
        depth_file_name = depth_folder_path + "/depth_" + imgs_index[i] + ".png"

        cam_tool.registration_existingimg(rgb_file_name, depth_file_name)

        # end_time = time.time()
        # execution_time = end_time - start_time
        # print("Execution time:", execution_time, "seconds")

        # print(feature_pixels_position[i, :, 0].tolist())

        # start_time = time.time()

        if feature_pixels_position[i, 0, 0] != -1:
            # print(feature_pixels_position[i, :, 0])
            # print(rgb_file_name)
            # feature_points_now = cam_tool.get_feature_points_3D(feature_pixels_position[i, :, 0].tolist(), 
            #                                                     feature_pixels_position[i, :, 1].tolist())
            feature_points_now = cam_tool.get_feature_points_3D(feature_pixels_position[i, :, 0].tolist(), 
                                                                feature_pixels_position[i, :, 1].tolist(), True)
            print(feature_points_now.shape)
            # print(type(feature_points_now))
            # pose_now = cam_tool.get_pose_6p()

            feature_points.append(feature_points_now)
            # pose.append(pose_now)
        else:
            feature_points_now = np.zeros((3, 12))
            feature_points_now[-1] = -1
            feature_points.append(feature_points_now)

        # end_time = time.time()
        # execution_time = end_time - start_time
        # print("Execution time:", execution_time, "seconds")

        # rgb = cam_tool.get_mat("rgb")
        # cv.imshow("rgb", rgb)
        

        depth_inrgb_CV8U = cam_tool.get_mat("depth_inrgb_CV8U")
        rgb_depth = cam_tool.get_mat("rgb_depth")
        rgb_drawn = cam_tool.get_mat("rgb_drawn")
        # depth_inrgb_CV8U = cam_tool.get_mat("depth_inrgb_CV8U")
        print(type(rgb_depth))
        cv.imshow("depth_inrgb_CV8U", depth_inrgb_CV8U)
        cv.imshow("rgb_depth", rgb_depth)
        cv.imshow("rgb_drawn", rgb_drawn)

        i += 1
        cv.waitKey(1)

        end_time = time.time()
        execution_time = end_time - start_time
        # print("Execution time:", execution_time, "seconds")


    # 结束后保存数据
    save_feature_points_path = "D:/aaaLab/aaagraduate/SaveVideo/scripts/total/ttt.txt"
    with open(save_feature_points_path, 'w') as f:
        for data in feature_points:
            # 四舍五入保留四位小数
            rounded_data = np.round(data, 4)
            cur_row = ''
            for i in range(6):
                x = rounded_data[0, i]
                y = rounded_data[1, i]
                z = rounded_data[2, i]
                s = '({:.4f}, {:.4f}, {:.4f})'.format(x, y, z)
                cur_row = cur_row + s + '\t'
            print(cur_row)


            # 格式化为字符串
            # formatted_data = ['({:.4f}, {:.4f}, {:.4f})'.format(x, y, z) for x, y, z in rounded_data]
            # print(formatted_data)
            # 将结果写入文件
            # f.write(' '.join(formatted_data) + '\n')
            f.write(''.join(cur_row) + '\n')

