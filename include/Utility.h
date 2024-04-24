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
	enum class FeatureID
	{
		LeftL = 37,
		LeftR = 40,
		RightL = 43,
		RightR = 46,
		NoseTop = 31,
		MouseTop = 52
	};

	static float PI;
	const int feature_num = 6;
	const std::vector<int> feature_id_series = { static_cast<int>(FeatureID::NoseTop),
												static_cast<int>(FeatureID::LeftL),
												static_cast<int>(FeatureID::LeftR),
												static_cast<int>(FeatureID::RightL),
												static_cast<int>(FeatureID::RightR),
												static_cast<int>(FeatureID::MouseTop),
	};


	Utility();
	~Utility();

	// ��ȡָ��·���ļ��У�ָ���к���λ�õ�Ԫ�أ�ʹ��ָ���ָ���
	std::string getValueAt(std::ifstream& file, int targetRow, int targetColumn, char delimiter);

	//auto GetFeaturePointsPixels(const std::string& feature_rgb_path, std::vector<std::vector<std::string>>& feature_pixels_position, char delimiter);

	// ��������ά�ַ�������洢Ϊtxt�ļ���ʹ��ָ���ָ���
	void saveToTxt(const std::vector<std::vector<std::string>>& data, const std::string& filename, char delimiter);
	//void saveToTxt(const std::vector<std::vector<float>>& data, const std::string& filename, char delimiter);
	//template<typename T>
	//void saveToTxt(const std::vector<std::vector<T>>& data, const std::string& filename, char delimiter);

	// ͨ���沿ָ���ĵ�λ�ã��õ�ͷ������λ��
	cv::Mat PositionToMotion(const cv::Mat& p1, const cv::Mat& p2, const cv::Mat& p3, const cv::Mat& p4);

	// ����ͷ���ľֲ�����ϵ
	void DrawCoord(cv::Mat& image, const cv::Mat& rot_mat, const cv::Mat& p1, const cv::Mat& p2, const cv::Mat& p3, const cv::Mat& p4);

	// ���
	void GetFeaturePointsPixels(const std::string& feature_rgb_path, std::vector<std::vector<std::string>>& feature_pixels_position, char delimiter);
	void GetFeaturePointsPixels_givenseries(const std::string& feature_rgb_path, std::vector<std::vector<std::string>>& feature_pixels_position, const std::vector<int>& id_series, char delimiter);

};

