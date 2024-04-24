"""
This is the algorithm to extract the feature point of people's face, for the final integrated system. 4/24
"""

import cv2
import dlib
import time

def cv_show(img,name):
    cv2.imshow('name',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def extract_feature_point(img):
#     img = cv2.resize(img, [int(img.shape[1] / 2), int(img.shape[0] / 2)])
#     # 转换为灰阶图片
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 正向人脸检测器将图像
#     detector = dlib.get_frontal_face_detector()
#     # 使用训练好的68个特征点模型
#     # predictor_path = "./shape_predictor_68_face_landmarks.dat"
#     predictor_path = "D:/aaaLab/aaagraduate/SaveVideo/DetectionApp/shape_predictor_68_face_landmarks.dat"
#     predictor = dlib.shape_predictor(predictor_path)
#     # 使用检测器来检测图像中的人脸
#     faces = detector(gray, 1)
#     # 打印结果
#     # print("Detected FaceNum: ", len(faces))
#     # win = dlib.image_window()
#     # win.set_image(img)
#     for j, face in enumerate(faces):
#         # print("Number ", j + 1, " Face Rectangular Box :/n/t", "left:", face.left(), "right:", face.right(), "top:",
#         #       face.top(), "bottom:", face.bottom())
#         imgC = img.copy()
#         imgT = img.copy()
#         cv2.rectangle(imgT,(face.left(),face.top()),(face.right(),face.bottom()),(0,0,255),1)
#         # 获取人脸特征点
#         shape = predictor(img, face)
#         points = []
#         for pt in shape.parts():
#             # 绘制特征点
#             pt_pos = (pt.x, pt.y)
#             cv2.circle(imgC, pt_pos, 1, (255, 0, 0), 2)
#             point_tmp = '(%d,%d)' % (pt.x, pt.y)
#             points.append(point_tmp)
#         Pts = []
#         for i in range(len(points)):
#             tmp1 = points[i].split(',')
#             tmp2 = int(tmp1[0][1:])
#             tmp3 = int(tmp1[1][:-1])
#             Pts.append(tuple([tmp2, tmp3]))
#         # imgT = drawFace(imgT,points)
#         # cv_show(imgC, imgC)
#         # win.add_overlay(shape)
#     # 绘制矩阵轮廓
#     # win.add_overlay(faces)
#     # dlib.hit_enter_to_continue()
#     return Pts,imgT


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


