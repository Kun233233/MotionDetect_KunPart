from PySide2.QtWidgets import QApplication, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import *
import cv2
import numpy as np
import dlib

# import time


import sys
import time
sys.path.append("D:/conda_envs/main-py38/Lib/site-packages/boost_python_cam")
import BoostPythonCam
# from scripts.total.all_test import GetCountMax, GetImgsIndex, GetPixels
from utils import extract_feature_point, GetCountMax, GetImgsIndex, GetPixels, save_points


# 主窗口类
class Capture:
    # 系统初始化
    def __init__(self):
        # 从文件中加载UI定义
        # 从UI定义中动态创建一个相应的窗口对象
        self.root_path = 'D:/aaaLab/aaagraduate/SaveVideo/DetectionApp/'
        self.ui = QUiLoader().load(self.root_path + 'ui/capture.ui')
        # self.ui = QUiLoader().load('ui/capture.ui')
        self.camera_timer = QTimer()
        # 获取显示器分辨率信息
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenHeight = self.screenRect.height()
        self.screenWidth = self.screenRect.width()
        # self.ui.setFixedSize(self.ui.width(), self.ui.height())
        self.record_frame = 0
        self.record_time = 0

        self.depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/DepthImgs1"
        self.rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/RGBImgs1"
        self.feature_rgb_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/1_68.txt"
        self.feature_3D_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/points1.txt"
        self.motion_vec_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/motion1.txt"
        self.save_points_12_path = "D:/aaaLab/aaagraduate/SaveVideo/scripts/total/ttt.txt"

        self.pixels_position = GetPixels(self.feature_rgb_path)
        self.feature_pixels_position = self.pixels_position[:, [22-1, 23-1, 30-1, 34-1, 37-1, 40-1, 43-1, 46-1, 49-1, 52-1, 55-1, 58-1], :]
        self.frame_path = self.rgb_folder_path + "/frame_num.txt"
        self.countmax = GetCountMax(self.frame_path) # 读取保存图片的数目，即帧数
        # self.countmax = 100 # 读取保存图片的数目，即帧数
        self.imgs_index = GetImgsIndex(self.countmax) # 生成对应帧的图片序列名
        self.count = 0 # 记录循环轮次
        self.feature_points = [] # 记录每一次循环特征点对应三维空间坐标
        self.feature_points_6 = []
        self.feature_points_12 = []
        self.width = 640
        self.height = 480
        self.predictor_path = "D:/aaaLab/aaagraduate/SaveVideo/DetectionApp/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.detector = dlib.get_frontal_face_detector()

    
    # 界面初始化
    def window_init(self):
        self.ui.OnButton.clicked.connect(self.open_camera) # 打开摄像头
        self.ui.OffButton.clicked.connect(self.close_camera) # 关闭摄像头
        self.camera_timer.timeout.connect(self.show_image)
    
    # 点击开始录制后操作
    def open_camera(self):
        # 设置摄像头
        # self.capture = cv2.VideoCapture(0)
        # 初始化RGBCamera
        camera_param_file_path = "D:/aaaLab/aaagraduate/SaveVideo/source/CameraParams.json"
        self.cam_tool = BoostPythonCam.RGBDCamera(camera_param_file_path, False, 0)
        # 检查摄像头是否打开
        # if not self.capture.isOpened():
        #     QMessageBox.critical(self.ui, '错误', '摄像头打开失败！')
        #     exit()
        # self.camera_timer.start(40)
        self.camera_timer.start(100)


    # 点击结束录制后操作
    def close_camera(self):
        self.camera_timer.stop()
        # self.capture.release()
        save_points(save_feature_points_path=self.save_points_12_path, feature_points=self.feature_points_12)
        self.childWindow = Curve()
        self.childWindow.ui.show()
    
    # 显示图片
    def show_image(self):
        start_time = time.time()
        # ret, self.image = self.capture.read()
        # 读取图像并配准
        rgb_file_name = self.rgb_folder_path + "/rgb_" + self.imgs_index[self.count] + ".png"
        depth_file_name = self.depth_folder_path + "/depth_" + self.imgs_index[self.count] + ".png"
        self.cam_tool.registration_existingimg(rgb_file_name, depth_file_name)

        # depth_inrgb_CV8U = self.cam_tool.get_mat("depth_inrgb_CV8U")
        # 获得rgb和深度图和融合图
        rgb_depth = self.cam_tool.get_mat("rgb_depth")
        rgb = self.cam_tool.get_mat("rgb")
        rgb_drawn = self.cam_tool.get_mat("rgb_drawn")
        # import pdb; pdb.set_trace()
        # start_time = time.time()
        Pts,imgT = extract_feature_point(rgb, self.detector, self.predictor)
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print("Execution time:", execution_time, "seconds")
        if len(Pts) == 0:
            Pts_np = np.full((68, 2), -1)
        else:
            Pts_np = np.array(Pts)*2
        Pts_12 = Pts_np[[22-1, 23-1, 30-1, 34-1, 37-1, 40-1, 43-1, 46-1, 49-1, 52-1, 55-1, 58-1], :]
        Pts_6 = Pts_np[[31-1, 37-1, 40-1, 43-1, 46-1, 52-1], :]
        # print(Pts_12[0])

        # print(rgb.shape)
        # if self.feature_pixels_position[self.count, 0, 0] != -1:
        #     feature_points_now = self.cam_tool.get_feature_points_3D(self.feature_pixels_position[self.count, :, 0].tolist(), 
        #                                                         self.feature_pixels_position[self.count, :, 1].tolist(), True)
        if Pts_12[0, 0] != -1:
            feature_points_now_6 = self.cam_tool.get_feature_points_3D(Pts_6[:, 0].tolist(), Pts_6[:, 1].tolist(), False)
            self.feature_points_6.append(feature_points_now_6)
            feature_points_now_12 = self.cam_tool.get_feature_points_3D(Pts_12[:, 0].tolist(), Pts_12[:, 1].tolist(), True)
            self.feature_points_12.append(feature_points_now_12)
        else:
            feature_points_now_6 = np.zeros((3, 6))
            feature_points_now_6[-1] = -1
            self.feature_points_6.append(feature_points_now_6)
            feature_points_now_12 = np.zeros((3, 12))
            feature_points_now_12[-1] = -1
            self.feature_points_12.append(feature_points_now_12)
            imgT = rgb
        # 检查是否正确获取图像
        # if not ret:
        #     QMessageBox.critical(self.ui, '错误', '无法获取帧！')
        #     exit()
        # width, height, _ = self.image.shape
        # self.image_show = cv2.flip(self.image, 1)
        # self.image_show = cv2.cvtColor(self.image_show, cv2.COLOR_BGR2RGB)
        # self.image_file = QImage(self.image_show.data, height, width, QImage.Format_RGB888)
        # self.image_show = cv2.flip(self.image, 1)
        t = cv2.cvtColor(rgb_depth, cv2.COLOR_BGR2RGB)
        rgb_depth_qt = QImage(t.data, self.width, self.height, QImage.Format_RGB888)
        t = cv2.cvtColor(rgb_drawn, cv2.COLOR_BGR2RGB)
        rgb_drawn_qt = QImage(t.data, self.width, self.height, QImage.Format_RGB888)
        t = cv2.cvtColor(imgT, cv2.COLOR_BGR2RGB)
        imgT_qt = QImage(t.data, self.width/2, self.height/2, QImage.Format_RGB888)

        self.ui.RGB_image.setPixmap(QPixmap.fromImage(rgb_drawn_qt))
        self.ui.depth_image.setPixmap(QPixmap.fromImage(rgb_depth_qt))
        # self.ui.landmark_image.setPixmap(QPixmap.fromImage(imgT_qt))
        # self.ui.xyz_image.setPixmap(QPixmap.fromImage(rgb_drawn_qt))
        self.ui.RGB_image.setScaledContents(True)
        self.ui.depth_image.setScaledContents(True)
        # self.ui.landmark_image.setScaledContents(True)
        # self.ui.xyz_image.setScaledContents(True)
        self.record_frame += 1
        self.record_time = int(self.record_frame/25)
        self.ui.statusbar.showMessage('已采集帧数 {} ，采集时间 {} s'.format(self.record_frame, self.record_time))

        self.count = self.count + 1

        end_time = time.time()
        execution_time = end_time - start_time
        # print("Execution time:", execution_time, "seconds")

        if self.count == self.countmax:
            self.close_camera()

# 子窗口类
class Curve:
    def __init__(self):
        self.root_path = 'D:/aaaLab/aaagraduate/SaveVideo/DetectionApp/'
        self.ui = QUiLoader().load(self.root_path + 'ui/curve.ui')

if __name__ == '__main__':
    app = QApplication([])
    save = Capture()
    save.window_init()
    save.ui.show()
    app.exit(app.exec_())