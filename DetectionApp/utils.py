"""
This is the algorithm to extract the feature point of people's face, for the final integrated system. 4/24
"""
import numpy as np
import cv2
import dlib
import time

def cv_show(img,name):
    cv2.imshow('name',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_feature_point(img, detector, predictor):
    img = cv2.resize(img, [int(img.shape[1] / 2), int(img.shape[0] / 2)])
    # 转换为灰阶图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 正向人脸检测器将图像
    # detector = dlib.get_frontal_face_detector()
    # 使用训练好的68个特征点模型
    # predictor_path = "./shape_predictor_68_face_landmarks.dat"
    # predictor_path = "D:/aaaLab/aaagraduate/SaveVideo/DetectionApp/shape_predictor_68_face_landmarks.dat"
    # predictor = dlib.shape_predictor(predictor_path)
    # 使用检测器来检测图像中的人脸
    # start_time = time.time()
    faces = detector(gray, 1) # about 0.02s
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("Execution time:", execution_time, "seconds")
    # 打印结果
    # print("Detected FaceNum: ", len(faces))
    # win = dlib.image_window()
    # win.set_image(img)
    points = []
    Pts = []
    imgT = None
    # start_time = time.time()
    for j, face in enumerate(faces):
        # print("Number ", j + 1, " Face Rectangular Box :/n/t", "left:", face.left(), "right:", face.right(), "top:",
        #       face.top(), "bottom:", face.bottom())
        imgC = img.copy()
        imgT = img.copy()
        cv2.rectangle(imgT,(face.left(),face.top()),(face.right(),face.bottom()),(0,0,255),1)
        # 获取人脸特征点
        shape = predictor(img, face)
        
        for pt in shape.parts():
            # 绘制特征点
            pt_pos = (pt.x, pt.y)
            cv2.circle(imgC, pt_pos, 1, (255, 0, 0), 2)
            point_tmp = '(%d,%d)' % (pt.x, pt.y)
            points.append(point_tmp)
        
        for i in range(len(points)):
            tmp1 = points[i].split(',')
            tmp2 = int(tmp1[0][1:])
            tmp3 = int(tmp1[1][:-1])
            # Pts.append(tuple([tmp2, tmp3]))
            Pts.append([tmp2, tmp3])
        # imgT = drawFace(imgT,points)
        # cv_show(imgC, imgC)
        # win.add_overlay(shape)
    # 绘制矩阵轮廓
    # win.add_overlay(faces)
    # dlib.hit_enter_to_continue()
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("Execution time:", execution_time, "seconds")

    return Pts,imgT


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

def save_points(save_feature_points_path, feature_points):
    # 结束后保存数据
    # save_feature_points_path = "D:/aaaLab/aaagraduate/SaveVideo/scripts/total/ttt.txt"
    with open(save_feature_points_path, 'w') as f:
        for data in feature_points:
            # 四舍五入保留四位小数
            rounded_data = np.round(data, 4)
            cur_row = ''
            for i in range(feature_points[0].shape[1]):
                x = rounded_data[0, i]
                y = rounded_data[1, i]
                z = rounded_data[2, i]
                s = '({:.4f}, {:.4f}, {:.4f})'.format(x, y, z)
                cur_row = cur_row + s + '\t'
            print(cur_row)

            f.write(''.join(cur_row) + '\n')

