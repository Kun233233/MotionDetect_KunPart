// class CameraModel, which is designed for computing pixels and points under the filed of camera.
#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/calib3d/calib3d.hpp>


class CameraModel
{
public:
	CameraModel();
	~CameraModel();

	void printMatrixInfo(const cv::Mat& matrix, const std::string& name);
	void hMirrorTrans(const cv::Mat& src, cv::Mat& dst);
	cv::Mat Get3DPoints(const cv::Mat& depth, const cv::Mat& pixels_to_points_map);
	cv::Mat GetPixels(const cv::Mat& points, const cv::Mat& camera_matrix, const cv::Mat& depth_map);

//protected:
	//RGB w x h
	const int IMAGE_WIDTH_640 = 640;
	const int  IMAGE_HEIGHT_480 = 480;
	//int max_depth; // �����ȡ��������

	float square_size; // ����궨ʱ���궨����ÿ�����̸�ı߳�����λ��mm


	// RGB��������
	cv::Mat RGBCameraMatrix;
	cv::Mat RGBCameraMatrix_inv;
	cv::Mat RGBDistCoeffs;
	cv::Mat RGBRotVec;
	cv::Mat RGBTransVec;
	cv::Mat RGBRotationMat;

	// �����������
	cv::Mat DepthCameraMatrix;
	cv::Mat DepthCameraMatrix_inv;
	cv::Mat DepthDistCoeffs;
	cv::Mat DepthRotVec;
	cv::Mat DepthTransVec;
	cv::Mat DepthRotationMat;

	// �������ͼ��ת��rgbͼ����׼������� equation: P_rgb = R * P_ir + T
	// Ŀǰ�Ȳ���һ��ͼ�����μ���ķ���
	// �������Ը�Ϊ���ö���ͼ��+��С���˷���С���
	cv::Mat R_depth2rgb;
	cv::Mat T_depth2rgb;



};