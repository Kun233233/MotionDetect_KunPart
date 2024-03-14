// class Utility, used in the project
#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/calib3d/calib3d.hpp>

#include <fstream>
#include <omp.h>


class Utility
{
public:
	static float PI;


	Utility();
	~Utility();

	// 获取指定路径文件中，指定行和列位置的元素，使用指定分隔符
	std::string getValueAt(std::ifstream& file, int targetRow, int targetColumn, char delimiter);

	//auto GetFeaturePointsPixels(const std::string& feature_rgb_path, std::vector<std::vector<std::string>>& feature_pixels_position, char delimiter);

	// 函数将二维字符串数组存储为txt文件，使用指定分隔符
	void saveToTxt(const std::vector<std::vector<std::string>>& data, const std::string& filename, char delimiter);

	// 通过面部指定四点位置，得到头部刚体位姿
	cv::Mat PositionToMotion(const cv::Mat& p1, const cv::Mat& p2, const cv::Mat& p3, const cv::Mat& p4);

	

};

