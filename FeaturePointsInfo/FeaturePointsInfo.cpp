
#include <iostream>
#include <OpenNI.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/calib3d/calib3d.hpp>

#include "OniSampleUtilities.h"
#include "UVC_Swapper.h"
#include "UVCSwapper.h"
#include "OBTypes.h"
#include "ObCommon.h"
//#include "OniSampleUtilities.h"

#include <fstream>
#include <sstream>
#include <regex>
#include <vector>
#include <chrono>
#include <cmath>

#include <omp.h>

//#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>

# include "CameraModel.h"
# include "Utility.h"


//RGB w x h
const int IMAGE_WIDTH_640 = 640;
const int  IMAGE_HEIGHT_480 = 480;
//Read data outtime
const int  UVC_TIME_OUT = 3000; //ms
const float PI = 3.1415926535;

const int feature_num = 6;


enum class FeatureID
{
	LeftL	= 37,
	LeftR	= 40,
	RightL	= 43,
	RightR	= 46, 
	NoseTop	= 34, 
	MouseTop= 52
};

const std::vector<int> feature_id_series = {	static_cast<int>(FeatureID::NoseTop),
												static_cast<int>(FeatureID::LeftL), 
												static_cast<int>(FeatureID::LeftR), 
												static_cast<int>(FeatureID::RightL), 
												static_cast<int>(FeatureID::RightR), 
												static_cast<int>(FeatureID::MouseTop),
												 };






void showdevice() {
	// 获取设备信息  
	openni::Array<openni::DeviceInfo> aDeviceList;
	openni::OpenNI::enumerateDevices(&aDeviceList);

	std::cout << "电脑上连接着 " << aDeviceList.getSize() << " 个体感设备." << endl;

	for (int i = 0; i < aDeviceList.getSize(); ++i)
	{
		cout << "设备 " << i << endl;
		const openni::DeviceInfo& rDevInfo = aDeviceList[i];
		cout << "设备名： " << rDevInfo.getName() << endl;
		cout << "设备Id： " << rDevInfo.getUsbProductId() << endl;
		cout << "供应商名： " << rDevInfo.getVendor() << endl;
		cout << "供应商Id: " << rDevInfo.getUsbVendorId() << endl;
		cout << "设备URI: " << rDevInfo.getUri() << endl;

	}
}

auto GetFeaturePointsPixels(const std::string& feature_rgb_path, std::vector<std::vector<std::string>>& feature_pixels_position, char delimiter)
{
	//// 打开文本文件
	std::ifstream file(feature_rgb_path);

	// 检查文件是否成功打开
	if (!file.is_open()) {
		std::cerr << "Error opening file" << std::endl;
		//return 1;
		exit(1);
	}


	std::string line;
	// 逐行读取文件内容
	for (int currentRow = 0; std::getline(file, line); ++currentRow) {
		// 使用字符串流解析每一行的数据
		std::istringstream iss(line);
		std::string value;
		std::vector<std::string> values;
		//std::vector<std::vector<std::string>> feature_pixels_position;
		if (currentRow == 0) { continue; }

		int currentColumn = 0;
		int nextFeature = 0;
		// 使用字段分隔符拆分每一行的数据
		while (std::getline(iss, value, delimiter)) {
			++currentColumn;

			// 如果当前列是目标列，则添加到 vector 中
			if (currentColumn == feature_id_series[nextFeature] + 1) {
				std::cout << currentRow - 1 << " " << currentColumn << " " << value << "\n";
				feature_pixels_position[nextFeature][currentRow - 1] = value;
				nextFeature++;
			}
		}


	}
	return;


}








int main()
{
	try {
		std::string  depth_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/depth_0.avi";
		std::string  rgb_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/rgb_0.avi";
		// 存储成图片形式的地址
		std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/DepthImgs";
		std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/RGBImgs";

		////std::ifstream file("D:/aaaLab/aaagraduate/SaveVideo/src/rgb.txt");
		std::regex pattern(R"(\((\d+),(\d+)\))");

		//// 打开文本文件
		//std::ifstream feature_rgb_file("D:/aaaLab/aaagraduate/SaveVideo/src/rgb.txt");
		std::string feature_rgb_path = "D:/aaaLab/aaagraduate/SaveVideo/source/rgb.txt";

		// 打开文件
		std::ifstream rgb_count_file(rgb_folder_path + "/frame_num.txt");
		// 检查文件是否成功打开
		if (!rgb_count_file.is_open()) {
			std::cerr << "Unable to open the file." << std::endl;
			return 1;
		}
		// 读取文件中的数字
		int countmax;
		rgb_count_file >> countmax;
		// 关闭文件
		rgb_count_file.close();
		
		CameraModel camera_model;
		Utility utility;

		// 创建一个 vector 用于存储每个元素
		std::vector<std::vector<std::string>> feature_pixels_position(feature_num);
		// 提前分配空间
		feature_pixels_position.resize(static_cast<int>(feature_num * 1.5));
		for (auto& row : feature_pixels_position) {
			row.resize(static_cast<int>(countmax * 1.5));
		}

		// 创建一个 vector 用于存储选中特征点的每一帧空间点坐标
		std::vector<std::vector<std::string>> feature_pixels_3D(feature_num);
		// 提前分配空间
		feature_pixels_3D.resize(static_cast<int>(feature_num * 1.5));
		for (auto& row : feature_pixels_3D) {
			row.resize(static_cast<int>(countmax * 1.5));
		}

		GetFeaturePointsPixels(feature_rgb_path, feature_pixels_position, '\t');

		// 计算像素到对应空间点坐标映射关系
		cv::Mat homogeneous_coords_all(3, IMAGE_HEIGHT_480 * IMAGE_WIDTH_640, CV_32F);
		homogeneous_coords_all.row(2).setTo(1);
		for (int y = 0; y < IMAGE_HEIGHT_480; ++y)
		{
			int column__ = y * IMAGE_WIDTH_640;
			for (int x = 0; x < IMAGE_WIDTH_640; ++x)
			{
				int column = column__ + x;
				homogeneous_coords_all.at<float>(0, column) = x;
				homogeneous_coords_all.at<float>(1, column) = y;
			}
		}
		cv::Mat pixels_to_points = camera_model.DepthCameraMatrix_inv * homogeneous_coords_all;

		// 记录循环次数
		int count = 0;

		

		double fps = 30.0;
		int wait_time = static_cast<int>(1 / fps * 1000.0);


		for (int i = 0; i < countmax; i++)
		{
			auto start_time = std::chrono::high_resolution_clock::now();

			std::string rgb_file_name = rgb_folder_path + "/rgb_" + std::to_string(i) + ".png"; 
			std::string depth_file_name = depth_folder_path + "/depth_" + std::to_string(i) + ".png";
			//std::cout << rgb_file_name << "\n";
			cv::Mat rgb = cv::imread(rgb_file_name);
			cv::Mat depth = cv::imread(depth_file_name, cv::IMREAD_UNCHANGED);
			//if (!depth.empty()) {
			//	std::cout << "Image size: " << depth.size() << std::endl;
			//	std::cout << "Number of channels: " << depth.channels() << std::endl;
			//}

			//else {
			//	std::cerr << "Failed to load image." << std::endl;
			//}
			//printMatrixInfo(depth, "depth");
			
			// 尝试一下畸变矫正
			//cv::Mat rgb_undistorted;
			//cv::undistort(rgb, rgb_undistorted, camera_model.RGBCameraMatrix, camera_model.RGBDistCoeffs);
			//cv::imshow("rgb_undistorted", rgb_undistorted);



			// 获得深度图每个像素点对应的3D空间坐标 (x, y, z)
			cv::Mat points = camera_model.Get3DPoints(depth, pixels_to_points);


			cv::Mat T_extended = cv::repeat(camera_model.T_depth2rgb, 1, IMAGE_HEIGHT_480 * IMAGE_WIDTH_640); // [3, 1] -> [3, 480*640]
			cv::Mat points_inrgb = camera_model.R_depth2rgb * points + T_extended; // points应该化成(3, 1)的样子，不急，回来改

			//cv::Mat depth_inrgb = GetPixels(points_inrgb, RGBCameraMatrix, hImageDepth);
			cv::Mat depth_inrgb = camera_model.GetPixels(points_inrgb, camera_model.RGBCameraMatrix, depth);

			// 中值滤波处理，先测试一下 (Kun: 2024.3.7)
			cv::medianBlur(depth_inrgb, depth_inrgb, 5);

			double max_depth_value;
			// 使用 cv::minMaxLoc 函数获取最大值和位置
			cv::minMaxLoc(depth_inrgb, nullptr, &max_depth_value, nullptr, nullptr);

			float scale_factor = 255.0 / static_cast<float>(max_depth_value);
			float offset = 0.0;
			cv::Mat depth_inrgb_CV8U;
			cv::convertScaleAbs(depth_inrgb, depth_inrgb_CV8U, scale_factor, offset);
			cv::imshow("depth_inrgb_CV8U", depth_inrgb_CV8U);


			// 将深度图归一化到0-255范围，以便与 RGB 图像叠加
			cv::Mat depth_inrgb_normalized;
			cv::normalize(depth_inrgb_CV8U, depth_inrgb_normalized, 0, 255, cv::NORM_MINMAX);


			// 将深度图转换为三通道，以便与 RGB 图像叠加
			cv::Mat depth_inrgb_color;
			cv::applyColorMap(depth_inrgb_normalized, depth_inrgb_color, cv::COLORMAP_JET);

			// 叠加深度图+rgb图像
			cv::Mat rgb_depth;
			double depthweight = 0.5;
			//cv::addWeighted(depth_inrgb_color, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
			cv::addWeighted(depth_inrgb_color, depthweight, rgb, (1 - depthweight), 0.0, rgb_depth);
			cv::imshow("Mixed", rgb_depth);

			// 显示帧
			//cv::imshow("Camera Feed", frame);
			//cv::imshow("Camera Feed", rgb);


			//std::ifstream file("D:/aaaLab/aaagraduate/SaveVideo/src/rgb.txt");
			//std::string value = getValueAt(file, i + 2, 37 + 1, '\t');
			//std::cout << value << "\n";
			//file.close();
			//std::string value = elements[i];
			std::vector<cv::Mat> position;
			position.resize(feature_num * 1.5);

			for (int feature_id = 0; feature_id < feature_num; feature_id++)
			{
				std::string value = feature_pixels_position[feature_id][i];
				
				std::cout << value << '\n';
				std::smatch matches;
				int x, y;
				if (std::regex_search(value, matches, pattern)) {
					// 第一个匹配项是整个字符串，后面的是括号内的两个数字
					x = std::stoi(matches[1].str());
					y = std::stoi(matches[2].str());
					std::cout << "First Number: " << x << std::endl;
					std::cout << "Second Number: " << y << std::endl;
				}
				else {
					std::cerr << "No match found" << std::endl;
				}
				//int index = y * IMAGE_WIDTH_640 + x;
				//int index = x * IMAGE_WIDTH_640 + y;
				//int index = x * IMAGE_HEIGHT_480 + y;
				int index = y * IMAGE_WIDTH_640 + x;
				// 检查索引是否在图像范围内
				if (index >= 0 && index < points_inrgb.cols) {
					// 访问 reshape 后的图像中特定位置的像素值
					float point_x = points_inrgb.at<float>(0, index);
					float point_y = points_inrgb.at<float>(1, index);
					float point_z = points_inrgb.at<float>(2, index);
					
					std::stringstream ss; // 创建一个字符串流对象
					ss << std::fixed << std::setprecision(4); // 设置小数点精度为4位
					ss << "(" << point_x << "," << point_y << "," << point_z << ")"; // 将浮点数写入字符串流中
					std::string result = ss.str(); // 从字符串流中获取组合后的字符串
					feature_pixels_3D[feature_id][i] = result;
					std::cout << result << std::endl; // 输出结果

					cv::Mat point = (cv::Mat_<float>(3, 1) << point_x, point_y, point_z);
					position[feature_id] = point;

					cv::Point feature(x, y);
					cv::circle(rgb, feature, 3, cv::Scalar(0, 0, 255), -1); // 红色点，半径为5
				}
				else {
					std::cerr << "Invalid index" << std::endl;
				}

			}
			cv::imshow("Camera Feed", rgb);

			cv::Mat motion = utility.PositionToMotion((position[1] + position[2]) / 2, (position[3] + position[4]) / 2, position[5], position[0]);


			auto end_time = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
			int new_wait_time = wait_time - static_cast<int>(duration.count());
			//std::cout << new_wait_time << " " << wait_time << " " << duration.count() << " " << static_cast<int>(duration.count());
			
			if (new_wait_time >= 1)
			{
				cv::waitKey(new_wait_time);
			}
			else
			{
				cv::waitKey(1);
			}
			


			// 终止快捷键 ESC
			if (cv::waitKey(1) == 27)
				break;

		}
		//file.close();
		std::string feature_3D_path = "D:/aaaLab/aaagraduate/SaveVideo/source/points.txt";
		utility.saveToTxt(feature_pixels_3D, feature_3D_path, '\t');

	}
	catch (cv::Exception& e)
	{
		std::cerr << "OpenCV Exception: " << e.what() << std::endl;
	}
	catch (std::exception& e)
	{
		std::cerr << "Standard Exception: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cerr << "An unknown exception occurred." << std::endl;
	}
	return 0;
}










