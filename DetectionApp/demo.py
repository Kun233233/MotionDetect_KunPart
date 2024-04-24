from PySide2.QtWidgets import QApplication, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import *
import cv2

# 主窗口类
class Capture:
    # 系统初始化
    def __init__(self):
        # 从文件中加载UI定义
        # 从UI定义中动态创建一个相应的窗口对象
        # self.ui = QUiLoader().load('ui/capture.ui')
        self.root_path = 'D:/aaaLab/aaagraduate/SaveVideo/DetectionApp/'
        self.ui = QUiLoader().load(self.root_path + 'ui/capture.ui')
        self.camera_timer = QTimer()
        # 获取显示器分辨率信息
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenHeight = self.screenRect.height()
        self.screenWidth = self.screenRect.width()
        self.record_frame = 0
        self.record_time = 0
    
    # 界面初始化
    def window_init(self):
        self.ui.OnButton.clicked.connect(self.open_camera) # 打开摄像头
        self.ui.OffButton.clicked.connect(self.close_camera) # 关闭摄像头
        self.camera_timer.timeout.connect(self.show_image)
    
    # 点击开始录制后操作
    def open_camera(self):
        # 设置摄像头
        self.capture = cv2.VideoCapture(0)
        # 检查摄像头是否打开
        if not self.capture.isOpened():
            QMessageBox.critical(self.ui, '错误', '摄像头打开失败！')
            exit()
        self.camera_timer.start(40)

    # 点击结束录制后操作
    def close_camera(self):
        self.camera_timer.stop()
        self.capture.release()
        self.childWindow = Curve()
        self.childWindow.ui.show()
    
    # 显示图片
    def show_image(self):
        ret, self.image = self.capture.read()
        # 检查是否正确获取图像
        if not ret:
            QMessageBox.critical(self.ui, '错误', '无法获取帧！')
            exit()
        width, height, _ = self.image.shape
        self.image_show = cv2.flip(self.image, 1)
        self.image_show = cv2.cvtColor(self.image_show, cv2.COLOR_BGR2RGB)
        self.image_file = QImage(self.image_show.data, height, width, QImage.Format_RGB888)
        self.ui.RGB_image.setPixmap(QPixmap.fromImage(self.image_file))
        self.ui.depth_image.setPixmap(QPixmap.fromImage(self.image_file))
        self.ui.landmark_image.setPixmap(QPixmap.fromImage(self.image_file))
        self.ui.xyz_image.setPixmap(QPixmap.fromImage(self.image_file))
        self.ui.RGB_image.setScaledContents(True)
        self.ui.depth_image.setScaledContents(True)
        self.ui.landmark_image.setScaledContents(True)
        self.ui.xyz_image.setScaledContents(True)
        self.record_frame += 1
        self.record_time = int(self.record_frame/25)
        self.ui.statusbar.showMessage('已采集帧数 {} ，采集时间 {} s'.format(self.record_frame, self.record_time))

# 子窗口类
class Curve:
    def __init__(self):
        self.root_path = 'D:/aaaLab/aaagraduate/SaveVideo/DetectionApp/'
        self.ui = QUiLoader().load(self.root_path + 'ui/curve.ui')
        # self.ui = QUiLoader().load('ui/curve.ui')

if __name__ == '__main__':
    app = QApplication([])
    save = Capture()
    save.window_init()
    save.ui.show()
    app.exit(app.exec_())