"""
This is the algorithm to extract the feature point of people's face, for the final integrated system. 4/24
"""
import numpy as np
import cv2
import dlib
import time
# import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
    rectangle_position = None
    for j, face in enumerate(faces):
        # print("Number ", j + 1, " Face Rectangular Box :/n/t", "left:", face.left(), "right:", face.right(), "top:",
        #       face.top(), "bottom:", face.bottom())
        imgC = img.copy()
        imgT = img.copy()
        cv2.rectangle(imgT,(face.left(),face.top()),(face.right(),face.bottom()),(0,0,255),1)
        rectangle_position = [face.left()*2, face.top()*2, face.right()*2, face.bottom()*2]
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

    return Pts, imgT, rectangle_position


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

'''
保存特征点对应三维空间坐标
'''
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
                s = '({:.4f},{:.4f},{:.4f})'.format(x, y, z)
                cur_row = cur_row + s + '\t'
            # print(cur_row)

            f.write(''.join(cur_row) + '\n')

'''
保存特征点对应像素坐标
'''
def save_pixels(save_feature_pixels_path, feature_pixels):
    # 结束后保存数据
    # save_feature_points_path = "D:/aaaLab/aaagraduate/SaveVideo/scripts/total/ttt.txt"
    with open(save_feature_pixels_path, 'w') as f:
        for data in feature_pixels:
            # 四舍五入保留四位小数
            rounded_data = np.round(data, 4)
            cur_row = ''
            for i in range(feature_pixels[0].shape[0]):
                x = rounded_data[i, 0]
                y = rounded_data[i, 1]
                s = '({:.4f},{:.4f})'.format(x, y)
                cur_row = cur_row + s + '\t'
            # print(cur_row)

            f.write(''.join(cur_row) + '\n')


def process(file_path, save_root_path):
    data_points = []

    with open(file_path, "r") as file:
        for line in file:
            point = [tuple(map(float, coord.strip('()').split(','))) for coord in line.strip().split()]
            data_points.append(point)

    array_3d = np.array(data_points)

    for row in range(array_3d.shape[0]):
        if np.all(np.all(array_3d[row, :, :] == array_3d[row, 0, :], axis=1)):
            array_3d[row, :, :] = [0.0, 0.0, -1.0]

    for row in range(array_3d.shape[0]):
        for col in range(array_3d.shape[1]):
            if np.array_equal(array_3d[row, col, :], [0.0, 0.0, -1.0]):
                up_index = row - 1
                while up_index >= 0:
                    if not np.array_equal(array_3d[up_index, col, :], [0.0, 0.0, -1.0]):
                        break
                    up_index -= 1
                down_index = row + 1
                while down_index < array_3d.shape[0]:
                    if not np.array_equal(array_3d[down_index, col, :], [0.0, 0.0, -1.0]):
                        break
                    down_index += 1

                if up_index >= 0 and down_index < array_3d.shape[0]:
                    up_value = array_3d[up_index, col, :]
                    down_value = array_3d[down_index, col, :]
                    distance = down_index - up_index
                    weight_up = (down_index - row) / distance
                    weight_down = (row - up_index) / distance
                    interpolated_value = weight_up * up_value + weight_down * down_value
                    array_3d[row, col, :] = interpolated_value

    n = len(array_3d)
    matrices = []
    for _ in range(n):
        matrix = np.zeros((3, 3))
        matrices.append(matrix)

    array_of_matrices = np.array(matrices)

    for i in range(n):
        array_of_matrices[i][1] = array_3d[i][0]+array_3d[i][1]+array_3d[i][5]+array_3d[i][6]-2*(array_3d[i][9]+array_3d[i][11])
        array_of_matrices[i][2] = np.cross(array_3d[i][0]+array_3d[i][5]-array_3d[i][9]-array_3d[i][11],array_3d[i][1]+array_3d[i][6]-array_3d[i][9]-array_3d[i][11])
    for i in range(n):
        array_of_matrices[i][1] = array_of_matrices[i][1]/np.linalg.norm(array_of_matrices[i][1])
        array_of_matrices[i][2] = array_of_matrices[i][2]/np.linalg.norm(array_of_matrices[i][2])
        array_of_matrices[i][0] = np.cross(array_of_matrices[i][1],array_of_matrices[i][2])

    transposed_matrices = np.array([np.transpose(matrix) for matrix in array_of_matrices])

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    for i in range(n):
        a[i]=transposed_matrices[i][2][1]/transposed_matrices[i][2][2]
        b[i]=-transposed_matrices[i][2][0]
        c[i]=transposed_matrices[i][1][0]/transposed_matrices[i][0][0]
    for i in range(n):
        a[i]=np.degrees(np.arctan(a[i]))
        b[i]=np.degrees(np.arcsin(b[i]))
        c[i]=np.degrees(np.arctan(c[i]))
    a_ini = np.mean(a[:15])
    b_ini = np.mean(b[:15])
    c_ini = np.mean(c[:15])
    for i in range(n):
        a[i]=a[i]-a_ini
        b[i]=b[i]-b_ini
        c[i]=c[i]-c_ini

    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    for i in range(n):
        # x[i]=(array_3d[i][2][0]+array_3d[i][3][0])/2+80*transposed_matrices[i][0][2]
        # y[i]=(array_3d[i][2][1]+array_3d[i][3][1])/2+80*transposed_matrices[i][1][2]
        # z[i]=(array_3d[i][2][2]+array_3d[i][3][2])/2+80*transposed_matrices[i][2][2]
        x[i]=array_3d[i][2][0]+80*transposed_matrices[i][0][2]
        y[i]=array_3d[i][2][1]+80*transposed_matrices[i][1][2]
        z[i]=array_3d[i][2][2]+80*transposed_matrices[i][2][2]
        # x[i]=(array_3d[i][2][0]+array_3d[i][3][0])/2
        # y[i]=(array_3d[i][2][1]+array_3d[i][3][1])/2
        # z[i]=(array_3d[i][2][2]+array_3d[i][3][2])/2
    x_ini = np.mean(x[:15])
    y_ini = np.mean(y[:15])
    z_ini = np.mean(z[:15])
    for i in range(n):
        x[i]=x[i]-x_ini
        y[i]=y[i]-y_ini
        z[i]=z[i]-z_ini

    rotation = np.zeros(n)
    motion = np.zeros(n)
    amplitude = np.zeros(n)
    for i in range(n):
        rotation[i]=np.sqrt(np.square(a[i])+np.square(b[i])+np.square(c[i]))
        motion[i]=np.sqrt(np.square(x[i])+np.square(y[i])+np.square(z[i]))
        amplitude[i] = 0.9*rotation[i]+0.1*motion[i]

    e, d = signal.butter(8, 0.1, 'lowpass')
    a = signal.filtfilt(e, d, a,axis=0)
    b = signal.filtfilt(e, d, b,axis=0)
    c = signal.filtfilt(e, d, c,axis=0)
    x = signal.filtfilt(e, d, x,axis=0)
    y = signal.filtfilt(e, d, y,axis=0)
    z = signal.filtfilt(e, d, z,axis=0)
    rotation = signal.filtfilt(e, d, rotation,axis=0)
    motion = signal.filtfilt(e, d, motion,axis=0)
    amplitude = signal.filtfilt(e, d, amplitude,axis=0)

    colors = ['#1f77b4' if x <= 10 else 'orange' if x <= 15 else 'red' for x in amplitude]
    # print(f"rotation: {rotation}")
    # print(f"motion: {motion}")
    # print(f"amplitude: {amplitude}")
    # # 合并为一个列表
    # combined_list = list(zip(rotation, motion, amplitude))
    # combined_list = [rotation, motion, amplitude]
    # # 将列表转换为 numpy 数组
    # data_array = np.array(combined_list)
    # # 保存为 txt 文件
    # np.savetxt(save_root_path + '/motion.txt', data_array, fmt='%d', delimiter='\t')
    with open(save_root_path + '/motion.txt', 'w') as f:
        for i in range(len(rotation)):
            # 四舍五入保留四位小数
            # rotation_rounded = np.round(rotation, 6)
            # motion_rounded = np.round(motion, 6)
            # amplitude_rounded = np.round(amplitude, 6)
            cur_row = "{:.6f}\t{:.6f}\t{:.6f}".format(rotation[i], motion[i], amplitude[i])
            f.write(''.join(cur_row) + '\n')


    plt.figure('合成角位移及平动位移')
    plt.subplot(211)
    plt.plot(rotation)
    plt.xlabel('Frame')
    plt.ylabel('Rotation(°)')
    plt.ylim((-5,55))
    plt.title('Synthetic Rotation')
    plt.subplot(212)
    plt.plot(motion)
    plt.xlabel('Frame')
    plt.ylabel('Displacement(mm)')
    plt.ylim((-5,55))
    plt.title('Synthetic Displacement')
    plt.tight_layout()
    plt.savefig(save_root_path + '/Rot_Trans.png')

    plt.figure('合成总位移')
    for i in range(n):
        plt.plot(i, amplitude[i], color=colors[i], marker='o', markersize=1.5)
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.ylim((-5,55))
    plt.title('Synthetic Motion')
    plt.savefig(save_root_path + '/Amplitude.png')
    plt.show()
    # return plt.figure('合成角位移及平动位移'), plt.figure('合成总位移'), amplitude

