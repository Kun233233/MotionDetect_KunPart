
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

#include <iostream>
//#include <pcl/point_cloud.h>
//#include <pcl/point_types.h>
//#include <pcl/common/common_headers.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/cloud_viewer.h>
//#include <pcl/console/parse.h>

# include "CameraModel.h"
# include "Utility.h"


//RGB w x h
const int IMAGE_WIDTH_640 = 640;
const int  IMAGE_HEIGHT_480 = 480;
//Read data outtime
const int  UVC_TIME_OUT = 3000; //ms
//const float PI = 3.1415926535;

//const int feature_num = 6;


//enum class FeatureID
//{
//	LeftL	= 37,
//	LeftR	= 40,
//	RightL	= 43,
//	RightR	= 46, 
//	NoseTop	= 31, 
//	MouseTop= 52
//};
//
//const std::vector<int> feature_id_series = {	static_cast<int>(FeatureID::NoseTop),
//												static_cast<int>(FeatureID::LeftL), 
//												static_cast<int>(FeatureID::LeftR), 
//												static_cast<int>(FeatureID::RightL), 
//												static_cast<int>(FeatureID::RightR), 
//												static_cast<int>(FeatureID::MouseTop),
//												 };
const std::vector<int> feature_id_series_12 = { 22, 23, 30, 34, 37, 40, 43, 46, 49, 52, 55, 58 };





void showdevice() {
	// 获取设备信息  
	openni::Array<openni::DeviceInfo> aDeviceList;
	openni::OpenNI::enumerateDevices(&aDeviceList);

	std::cout << "电脑上连接着 " << aDeviceList.getSize() << " 个体感设备." << std::endl;

	for (int i = 0; i < aDeviceList.getSize(); ++i)
	{
		std::cout << "设备 " << i << std::endl;
		const openni::DeviceInfo& rDevInfo = aDeviceList[i];
		std::cout << "设备名： " << rDevInfo.getName() << std::endl;
		std::cout << "设备Id： " << rDevInfo.getUsbProductId() << std::endl;
		std::cout << "供应商名： " << rDevInfo.getVendor() << std::endl;
		std::cout << "供应商Id: " << rDevInfo.getUsbVendorId() << std::endl;
		std::cout << "设备URI: " << rDevInfo.getUri() << std::endl;

	}
}









int main()
{
	try {
		double minVal, maxVal;
		//std::string  depth_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/depth_0.avi";
		//std::string  rgb_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/rgb_0.avi";


		// 存储成图片形式的地址

		// -------------------
		//std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/DepthImgs"; // 深度图文件夹路径
		//std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/RGBImgs"; // rgb文件夹路径
		//std::string feature_rgb_path = "D:/aaaLab/aaagraduate/SaveVideo/source/rgb.txt"; // 特征点像素索引txt文件路径
		//std::string feature_3D_path = "D:/aaaLab/aaagraduate/SaveVideo/source/points.txt"; // 存储特征点三维空间坐标
		//std::string motion_vec_path = "D:/aaaLab/aaagraduate/SaveVideo/source/motion.txt"; // 存储运动信息
		// -------------------

		// ------------------- 20240314 1 ------------------- 
		//std::string  depth_folder_path ="D:/aaaLab/aaagraduate/SaveVideo/source/20240314/DepthImgs1";
		//std::string  rgb_folder_path =	"D:/aaaLab/aaagraduate/SaveVideo/source/20240314/RGBImgs1";
		//std::string feature_rgb_path =	"D:/aaaLab/aaagraduate/SaveVideo/source/20240314/0312_68_01.txt";
		//std::string feature_3D_path =	"D:/aaaLab/aaagraduate/SaveVideo/source/20240314/points1.txt";
		//std::string feature_3D_path_12 =	"D:/aaaLab/aaagraduate/SaveVideo/source/20240314/points1_12.txt";
		//std::string motion_vec_path =	"D:/aaaLab/aaagraduate/SaveVideo/source/20240314/motion1.txt";
		// ------------------- end ------------------- 

		// ------------------- 20240314 2 ------------------- 
		//std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/DepthImgs2";
		//std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/RGBImgs2";
		//std::string feature_rgb_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/0312_68_02.txt";
		//std::string feature_3D_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/points2.txt";
		//std::string feature_3D_path_12 = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/points2_12.txt";
		//std::string motion_vec_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/motion2.txt";
		// ------------------- end ------------------- 

		// ------------------- 20240326 1 ------------------- 
		std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/DepthImgs1";
		std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/RGBImgs1";
		std::string feature_rgb_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/1_68.txt";
		std::string feature_3D_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/points1.txt";
		std::string feature_3D_path_12 = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/points1_12.txt";
		std::string motion_vec_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/motion1.txt";
		// ------------------- end ------------------- 

		// ------------------- 20240326 2 ------------------- 
		//std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/DepthImgs2";
		//std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/RGBImgs2";
		//std::string feature_rgb_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/2_68.txt";
		//std::string feature_3D_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/points2.txt";
		//std::string feature_3D_path_12 = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/points2_12.txt";
		//std::string motion_vec_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/motion2.txt";
		// ------------------- end ------------------- 

		// ------------------- 20240326 3 ------------------- 
		//std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/DepthImgs3";
		//std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/RGBImgs3";
		//std::string feature_rgb_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/3_68.txt";
		//std::string feature_3D_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/points3.txt";
		//std::string feature_3D_path_12 = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/points3_12.txt";
		//std::string motion_vec_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/motion3.txt";
		// ------------------- end ------------------- 





		////std::ifstream file("D:/aaaLab/aaagraduate/SaveVideo/src/rgb.txt");
		std::regex pattern(R"(\((\d+),(\d+)\))");

		//// 打开文本文件
		//std::ifstream feature_rgb_file("D:/aaaLab/aaagraduate/SaveVideo/src/rgb.txt");
		//std::string feature_rgb_path = "D:/aaaLab/aaagraduate/SaveVideo/source/rgb.txt";

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

		// 生成图片文件序号
		std::vector<std::string> file_index;
		for (int i = 0; i < countmax; i++)
		{
			std::string name = std::to_string(i);
			int t = name.length(); //存储当前序号的数字位数
			
			for (int j = 0; j < 6 - t; j++)
			{
				name = "0" + name;
			}
			file_index.push_back(name);
			std::cout << name << '\n';
		}



		
		//CameraModel camera_model;
		std::string camera_params_file_path = "D:/aaaLab/aaagraduate/SaveVideo/source/CameraParams.json";
		CameraModel camera_model(camera_params_file_path);
		Utility utility;

		// 创建一个 vector 用于存储每个元素
		std::vector<std::vector<std::string>> feature_pixels_position;
		// 提前分配空间
		feature_pixels_position.resize(static_cast<int>(utility.feature_num));
		for (auto& row : feature_pixels_position) {
			row.resize(static_cast<int>(countmax));
		}

		// 创建一个 vector 用于存储选中特征点的每一帧空间点坐标
		std::vector<std::vector<std::string>> feature_pixels_3D;
		// 提前分配空间
		feature_pixels_3D.resize(static_cast<int>(countmax));
		for (auto& row : feature_pixels_3D) {
			row.resize(static_cast<int>(utility.feature_num));
		}

		// 创建一个 vector 用于存储每个元素，12个特征点
		std::vector<std::vector<std::string>> feature_pixels_position_12;
		// 提前分配空间
		feature_pixels_position_12.resize(static_cast<int>(12));
		for (auto& row : feature_pixels_position_12) {
			row.resize(static_cast<int>(countmax));
		}

		// 创建一个 vector 用于存储选中特征点的每一帧空间点坐标，12个特征点
		std::vector<std::vector<std::string>> feature_pixels_3D_12;
		// 提前分配空间
		feature_pixels_3D_12.resize(static_cast<int>(countmax));
		for (auto& row : feature_pixels_3D_12) {
			row.resize(static_cast<int>(12));
		}

		// motion_vec 存储每一帧的运动情况
		std::vector<std::vector<std::string>> motion_vec;
		// 提前分配空间
		motion_vec.resize(static_cast<int>(countmax));
		for (auto& row : motion_vec) {
			row.resize(static_cast<int>(6));
		}

		utility.GetFeaturePointsPixels(feature_rgb_path, feature_pixels_position, '\t');
		utility.GetFeaturePointsPixels_givenseries(feature_rgb_path, feature_pixels_position_12, feature_id_series_12, '\t');

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

		//// 初始化变量来保存最大和最小值以及它们的位置
		//double minValue, maxValue;
		//cv::Point minLoc, maxLoc;
		//// 使用cv::minMaxLoc函数获取最大和最小值及其位置
		//cv::minMaxLoc(homogeneous_coords_all, &minValue, &maxValue, &minLoc, &maxLoc);
		//// 输出结果
		//std::cout << "Min value: " << minValue << ", Max value: " << maxValue << std::endl;
		//std::cout << "Min location: " << minLoc << ", Max location: " << maxLoc << std::endl;

		//cv::Mat points_x = pixels_to_points.row(0).reshape(1, 480);
		//cv::Mat points_y = pixels_to_points.row(1).reshape(1, 480);
		//cv::Mat n0;
		//cv::Mat n1;
		//cv::normalize(points_x, n0, 0, 1, cv::NORM_MINMAX);
		//cv::normalize(points_y, n1, 0, 1, cv::NORM_MINMAX);
		//cv::imshow("n0", n0);
		//cv::imshow("n1", n1);

		// 记录原始depth中为0的点在rgb坐标系下对应的坐标，便于剔除异常数据
		cv::Mat nodepth_point = (cv::Mat_<float>(3, 1) << 0.0f, 0.0f, 0.0f);
		cv::Mat nodepth_point_inrgb = camera_model.R_depth2rgb * nodepth_point + camera_model.T_depth2rgb;
		std::cout << nodepth_point_inrgb << '\n';



		// 记录循环次数
		int count = 0;

		

		double fps = 30.0;
		int wait_time = static_cast<int>(1 / fps * 1000.0);


		for (int i = 0; i < countmax; i++)
		{
			auto start_time = std::chrono::high_resolution_clock::now();

			//std::string rgb_file_name = rgb_folder_path + "/rgb_" + std::to_string(i) + ".png"; 
			//std::string depth_file_name = depth_folder_path + "/depth_" + std::to_string(i) + ".png";
			std::string rgb_file_name = rgb_folder_path + "/rgb_" + file_index[i] + ".png";
			std::string depth_file_name = depth_folder_path + "/depth_" + file_index[i] + ".png";
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
			auto end_time = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
			std::cout << "duration: " << duration.count() << '\n';

			// 获得深度图每个像素点对应的3D空间坐标 (x, y, z)
			cv::Mat points = camera_model.Get3DPoints(depth, pixels_to_points);
			//cv::Mat points_x = points.row(0).reshape(1, 480);
			//cv::Mat points_y = points.row(1).reshape(1, 480);
			cv::Mat points_z = points.row(2).reshape(1, 480);
			//cv::Mat n;
			//cv::normalize(points_x, n, 0, 1, cv::NORM_MINMAX);
			//cv::imshow("n", n);
			//cv::minMaxLoc(points_z, &minVal, &maxVal);
			//std::cout << minVal << " " << maxVal << '\n';

			



			cv::Mat T_extended = cv::repeat(camera_model.T_depth2rgb, 1, IMAGE_HEIGHT_480 * IMAGE_WIDTH_640); // [3, 1] -> [3, 480*640]
			cv::Mat points_inrgb = camera_model.R_depth2rgb * points + T_extended; // points应该化成(3, 1)的样子，不急，回来改

			//cv::minMaxLoc(points_inrgb.row(2).reshape(1, 480), &minVal, &maxVal);
			//std::cout << minVal << " " << maxVal << '\n';

			//cv::Mat points_inrgb_x = points_inrgb.row(0).reshape(1, 480);
			//cv::Mat points_inrgb_y = points_inrgb.row(1).reshape(1, 480);
			//cv::Mat points_inrgb_z = points_inrgb.row(2).reshape(1, 480);
			//cv::Mat n_inrgb;
			//cv::normalize(points_x, n_inrgb, 0, 1, cv::NORM_MINMAX);
			//cv::imshow("n_inrgb", n_inrgb);
			//cv::waitKey(0);

			//for (int j = 0; j < 640 * 480; j++)
			//{
			//	if (std::abs(points_inrgb.at<float>(0, j) - (-24.6678)) < 1e-3)
			//	{
			//		//std::cout << x << " " << y << '\n';
			//		std::cout << "----------" << '\n';
			//		std::cout << points.at<float>(0, j) << " " << points.at<float>(1, j) << " " << points.at<float>(2, j) << " " << '\n';
			//		std::cout << "----------" << '\n';
			//		cv::waitKey(0);
			//	}
			//}
			

			//cv::Mat depth_inrgb = GetPixels(points_inrgb, RGBCameraMatrix, hImageDepth);
			cv::Mat depth_inrgb = camera_model.GetPixels(points_inrgb, camera_model.RGBCameraMatrix, depth);
			
			// 转换至rgb坐标系下，方便构建点云
			cv::Mat points_rgbcoord = camera_model.PixelsCoordTransfer(points_inrgb);

			cv::medianBlur(points_rgbcoord, points_rgbcoord, 5); // =============

			//// 分离通道
			//std::vector<cv::Mat> channels;
			//cv::split(points_rgbcoord, channels);

			//// 获取要可视化的通道索引（0: Blue, 1: Green, 2: Red）
			//int channelIndex = 0; // 这里选择蓝色通道

			////// 将大于阈值的元素设为0
			////cv::threshold(channels[channelIndex], channels[channelIndex], 1000, 0, cv::THRESH_TOZERO);
			////// 将小于阈值的元素设为0
			////cv::threshold(channels[channelIndex], channels[channelIndex], -1000, 0, cv::THRESH_TOZERO_INV);

			//// 初始化变量来保存最大和最小值以及它们的位置
			//double minValue, maxValue;
			//cv::Point minLoc, maxLoc;
			//

			//// 使用cv::minMaxLoc函数获取最大和最小值及其位置
			//cv::minMaxLoc(channels[channelIndex], &minValue, &maxValue, &minLoc, &maxLoc);

			//// 输出结果
			//std::cout << "Min value: " << minValue << ", Max value: " << maxValue << std::endl;
			//std::cout << "Min location: " << minLoc << ", Max location: " << maxLoc << std::endl;

			//// 归一化选定通道到 0-255 范围
			//cv::Mat normalizedChannel;
			//cv::normalize(channels[channelIndex], normalizedChannel, 0, 1, cv::NORM_MINMAX);

			//// 可视化选定通道
			//cv::imshow("Channel", normalizedChannel);
			//cv::waitKey(0);

			//std::cout << "34343434" << '\n';

			// 中值滤波处理，先测试一下 (Kun: 2024.3.7)
			cv::medianBlur(depth_inrgb, depth_inrgb, 5); // =============
			cv::medianBlur(points_rgbcoord, points_rgbcoord, 5); // =============


			// 寻找最小值和最大值
			cv::minMaxLoc(depth_inrgb, &minVal, &maxVal);
			//std::cout << minVal << " " << maxVal << '\n';

			//camera_model.printMatrixInfo(depth_inrgb, "depth_inrgb");

			// 线性变换将floatMat的值缩放到0-255的范围内
			cv::Mat scaledMat;
			float scale =  255.0 / static_cast<float>((maxVal - minVal));
			float shift = static_cast<float>(-minVal) * scale;
			
			cv::Mat depth_inrgb_CV8U;
			//cv::convertScaleAbs(depth_inrgb, depth_inrgb_CV8U, scale_factor, offset);
			depth_inrgb.convertTo(depth_inrgb_CV8U, CV_8U, scale, shift);
			//cv::minMaxLoc(depth_inrgb, &minVal, &maxVal);
			//std::cout << minVal << " " << maxVal << '\n';

			//cv::normalize(depth_inrgb, depth_inrgb_CV8U, 0, 1, cv::NORM_MINMAX);
			cv::imshow("depth_inrgb_CV8U", depth_inrgb_CV8U); ////
			//cv::waitKey(0);



			// 将深度图归一化到0-255范围，以便与 RGB 图像叠加
			//cv::Mat depth_inrgb_normalized;
			//cv::normalize(depth_inrgb_CV8U, depth_inrgb_normalized, 0, 255, cv::NORM_MINMAX);


			// 将深度图转换为三通道，以便与 RGB 图像叠加
			cv::Mat depth_inrgb_color; 
			//cv::applyColorMap(depth_inrgb_normalized, depth_inrgb_color, cv::COLORMAP_JET);
			cv::applyColorMap(depth_inrgb_CV8U, depth_inrgb_color, cv::COLORMAP_JET);


			// 叠加深度图+rgb图像
			cv::Mat rgb_depth;
			double depthweight = 0.5;
			//cv::addWeighted(depth_inrgb_color, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
			cv::addWeighted(depth_inrgb_color, depthweight, rgb, (1 - depthweight), 0.0, rgb_depth);
			cv::imshow("Mixed", rgb_depth); ////

			// 显示帧
			//cv::imshow("Camera Feed", frame);
			//cv::imshow("Camera Feed", rgb);


			//std::ifstream file("D:/aaaLab/aaagraduate/SaveVideo/src/rgb.txt");
			//std::string value = getValueAt(file, i + 2, 37 + 1, '\t');
			//std::cout << value << "\n";
			//file.close();
			//std::string value = elements[i];
			std::vector<cv::Mat> position;
			position.resize(utility.feature_num * 1.5);

			bool is_empty = false;

			for (int feature_id = 0; feature_id < utility.feature_num; feature_id++)
			{
				std::string value = feature_pixels_position[feature_id][i];
				
				//std::cout << value << '\n';
				std::smatch matches;
				int x, y;
				if (std::regex_search(value, matches, pattern)) {
					// 第一个匹配项是整个字符串，后面的是括号内的两个数字
					x = std::stoi(matches[1].str());
					y = std::stoi(matches[2].str());
					//std::cout << "First Number: " << x << std::endl;
					//std::cout << "Second Number: " << y << std::endl;
				}
				else {
					std::cerr << "No match found" << std::endl;
				}

				//int index = y * IMAGE_WIDTH_640 + x;
				// 检查索引是否在图像范围内
				//if (index >= 0 && index < points_rgbcoord.cols) {
				if (true){
					// 访问 reshape 后的图像中特定位置的像素值
					float point_x = points_rgbcoord.at<cv::Vec3f>(y, x)[0];
					float point_y = points_rgbcoord.at<cv::Vec3f>(y, x)[1];
					float point_z = points_rgbcoord.at<cv::Vec3f>(y, x)[2];

					//std::vector<cv::Mat> channels;
					//cv::split(points_rgbcoord, channels);
					//cv::Point feature1(x, y);
					//cv::normalize(channels[0], channels[0], 0, 1, cv::NORM_MINMAX);
					//cv::circle(channels[0], feature1, 3, cv::Scalar(255), -1);
					//cv::imshow("channels[0]", channels[0]);



					//if (std::abs(point_x - (-24.6678)) < 1e-3)
					//{
					//	std::cout << x << " " << y << '\n';
					//	std::cout << depth_inrgb.at<float>(y, x) << '\n';
					//	//cv::waitKey(0);
					//}
					
					//std::cout << y << " " << x << " " << depth_inrgb.at<uint16_t>(x, y) << '\n';
					//std::cout << nodepth_point_inrgb.at<float>(2, 0) << '\n';
					if (std::abs(point_z - nodepth_point_inrgb.at<float>(2, 0)) < 1e-4)
					{
						std::cout << x << " " << y << '\n';
						std::cout << depth_inrgb.at<float>(y, x) << '\n';
						point_x = 0.0f;
						point_y = 0.0f;
						point_z = -1.0f;	
						is_empty = true;
					}

					
					std::stringstream ss; // 创建一个字符串流对象
					ss << std::fixed << std::setprecision(4); // 设置小数点精度为4位
					ss << "(" << point_x << "," << point_y << "," << point_z << ")"; // 将浮点数写入字符串流中
					std::string result = ss.str(); // 从字符串流中获取组合后的字符串
					feature_pixels_3D[i][feature_id] = result;
					//std::cout << result << std::endl; // 输出结果
					//std::cout << feature_pixels_3D.size() << "  " << feature_pixels_3D[feature_id].size() << std::endl;

					cv::Mat point = (cv::Mat_<float>(3, 1) << point_x, point_y, point_z);
					position[feature_id] = point;

					cv::Point feature(x, y);
					cv::circle(rgb, feature, 3, cv::Scalar(0, 0, 255), -1); // 红色点，半径为5
				}
				else {
					std::cerr << "Invalid index" << std::endl;
				}

			}


			
			// ----------------------------------- 保存孙喆提出要求的12个特征点 ----------------------------------- 
			for (int feature_id = 0; feature_id < 12; feature_id++)
			{
				std::string value = feature_pixels_position_12[feature_id][i];
				std::smatch matches;
				int x, y;
				if (std::regex_search(value, matches, pattern)) {
					// 第一个匹配项是整个字符串，后面的是括号内的两个数字
					x = std::stoi(matches[1].str());
					y = std::stoi(matches[2].str());
				}
				else {
					std::cerr << "No match found" << std::endl;
				}

				// 访问 reshape 后的图像中特定位置的像素值
				float point_x = points_rgbcoord.at<cv::Vec3f>(y, x)[0];
				float point_y = points_rgbcoord.at<cv::Vec3f>(y, x)[1];
				float point_z = points_rgbcoord.at<cv::Vec3f>(y, x)[2];

				if (std::abs(point_z - nodepth_point_inrgb.at<float>(2, 0)) < 1e-4)
				{
					std::cout << x << " " << y << '\n';
					std::cout << depth_inrgb.at<float>(y, x) << '\n';
					point_x = 0.0f;
					point_y = 0.0f;
					point_z = -1.0f;
					is_empty = true;
				}


				std::stringstream ss; // 创建一个字符串流对象
				ss << std::fixed << std::setprecision(4); // 设置小数点精度为4位
				ss << "(" << point_x << "," << point_y << "," << point_z << ")"; // 将浮点数写入字符串流中
				std::string result = ss.str(); // 从字符串流中获取组合后的字符串
				feature_pixels_3D_12[i][feature_id] = result;

			}
			// ----------------------------------- end -----------------------------------
			std::cout << 1111 << '\n';
			// ----------------------------------- 新的位姿计算 -----------------------------------  
			cv::Mat motion(6, 1, CV_32F);
			if (is_empty) // 如果存在异常数据，直接全部赋值为-1
			{
				motion.at<float>(0, 0) = -1.0f;
				motion.at<float>(1, 0) = -1.0f;
				motion.at<float>(2, 0) = -1.0f;
				motion.at<float>(3, 0) = -1.0f;
				motion.at<float>(4, 0) = -1.0f;
				motion.at<float>(5, 0) = -1.0f;

			}
			else // 正常处理
			{
				cv::Mat p1 = (position[1] + position[2]) / 2;
				cv::Mat p2 = (position[3] + position[4]) / 2;
				cv::Mat p3 = position[5];
				cv::Mat p4 = position[0];

				cv::Mat motion = utility.PositionToMotion(p1, p2, p3, p4);
				utility.DrawCoord(rgb, camera_model.RGBCameraMatrix, p1, p2, p3, p4);
				for (int motion_id = 0; motion_id < 6; motion_id++)
				{
					motion_vec[i][motion_id] = std::to_string(motion.at<float>(motion_id, 0));
				}

			}
			cv::imshow("Camera Feed", rgb);
			// ----------------------------------- end -----------------------------------

			////cv::imshow("Camera Feed", rgb);
			////std::cout << camera_model.RGBRotationMat << "\n";
			//cv::Mat motion = utility.PositionToMotion((position[1] + position[2]) / 2, (position[3] + position[4]) / 2, position[5], position[0]);
			//
			//for (int motion_id = 0; motion_id < 6; motion_id++)
			//{
			//	if (!is_empty)
			//	{
			//		motion_vec[i][motion_id] = std::to_string( motion.at<float>(motion_id, 0));
			//		utility.DrawCoord(rgb, camera_model.RGBCameraMatrix, (position[1] + position[2]) / 2, (position[3] + position[4]) / 2, position[5], position[0]);
			//	}
			//	else
			//	{
			//		motion_vec[i][motion_id] = "";
			//	}
			//	
			//}
			//cv::imshow("Camera Feed", rgb); ////

			end_time = std::chrono::high_resolution_clock::now();
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
			//auto end_time = std::chrono::high_resolution_clock::now();
			//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
			int new_wait_time = wait_time - static_cast<int>(duration.count());
			//std::cout << new_wait_time << " " << wait_time << " " << duration.count() << " " << static_cast<int>(duration.count()) << '\n';
			
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
		//std::string feature_3D_path = "D:/aaaLab/aaagraduate/SaveVideo/source/points.txt";
		utility.saveToTxt(feature_pixels_3D, feature_3D_path, '\t');
		//utility.saveToTxt<std::string>(feature_pixels_3D, feature_3D_path, '\t');

		//std::string motion_vec_path = "D:/aaaLab/aaagraduate/SaveVideo/source/motion.txt";
		utility.saveToTxt(motion_vec, motion_vec_path, '\t');
		//utility.saveToTxt<float>(motion_vec, motion_vec_path, '\t');
		//std::string feature_3D_path = "D:/aaaLab/aaagraduate/SaveVideo/source/points.txt";
		utility.saveToTxt(feature_pixels_3D_12, feature_3D_path_12, '\t');

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










