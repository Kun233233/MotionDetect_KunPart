// class CameraModel, which is designed for computing pixels and points under the filed of camera.
#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/calib3d/calib3d.hpp>

# include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>


class CameraModel
{
public:
	CameraModel();
	CameraModel(std::string file_path);
	~CameraModel();

	void printMatrixInfo(const cv::Mat& matrix, const std::string& name);
	void hMirrorTrans(const cv::Mat& src, cv::Mat& dst);
	cv::Mat Get3DPoints(const cv::Mat& depth, const cv::Mat& pixels_to_points_map);
	cv::Mat GetPixels(const cv::Mat& points, const cv::Mat& camera_matrix, const cv::Mat& depth_map);

	cv::Mat PixelsCoordTransfer(const cv::Mat& points);

//protected:
	//RGB w x h
	const int IMAGE_WIDTH_640 = 640;
	const int  IMAGE_HEIGHT_480 = 480;
	//int max_depth; // 相机获取的最大深度

	// 存储图片高和宽，这里写的不是太好，有机会再改吧
	int height = 480;
	int width = 640;

	float square_size; // 相机标定时，标定板中每个棋盘格的边长，单位：mm


	// RGB相机内外参
	cv::Mat RGBCameraMatrix;
	cv::Mat RGBCameraMatrix_inv;
	cv::Mat RGBDistCoeffs; // 畸变参数的一般顺序是k1,k2,p1,p2,k3
	cv::Mat RGBRotVec;
	cv::Mat RGBTransVec;
	cv::Mat RGBRotationMat;

	// 深度相机内外参
	cv::Mat DepthCameraMatrix;
	cv::Mat DepthCameraMatrix_inv;
	cv::Mat DepthDistCoeffs;
	cv::Mat DepthRotVec;
	cv::Mat DepthTransVec;
	cv::Mat DepthRotationMat;

	// 计算深度图像转到rgb图像配准所需参数 equation: P_rgb = R * P_ir + T
	// 目前先采用一个图像的外参计算的方法
	// 后续可以改为利用多组图像+最小二乘法减小误差
	cv::Mat R_depth2rgb;
	cv::Mat T_depth2rgb;

	// 将原始points对应像素位置（也就是深度图中points对应像素位置），转换至rgb图像下像素坐标，目的是方便点云生成
	cv::Mat map_x;
	cv::Mat map_y;


	cv::Mat depth_pixels_to_points;



};