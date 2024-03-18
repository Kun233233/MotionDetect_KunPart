//#include <iostream>
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>
//#include <pcl/visualization/pcl_visualizer.h>
//
//int main(int argc, char** argv)
//{
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//
//    // Fill in the cloud data
//    cloud->width = 5;
//    cloud->height = 1;
//    cloud->is_dense = false;
//    cloud->points.resize(cloud->width * cloud->height);
//
//    for (auto& point : *cloud)
//    {
//        point.x = 1024 * rand() / (RAND_MAX + 1.0f);
//        point.y = 1024 * rand() / (RAND_MAX + 1.0f);
//        point.z = 1024 * rand() / (RAND_MAX + 1.0f);
//    }
//
//    pcl::io::savePCDFileASCII("test_pcd.pcd", *cloud);
//    std::cerr << "Saved " << cloud->size() << " data points to test_pcd.pcd." << std::endl;
//
//    // 创建 PCL 可视化窗口
//    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("PointCloud Viewer"));
//    viewer->setBackgroundColor(0.0, 0.0, 0.0);
//
//    // 将点云添加到可视化窗口
//    viewer->addPointCloud(cloud, "point_cloud");
//
//    // 显示可视化窗口
//    viewer->spin();
//
//    return (0);
//}
//
//
//
//
//
//
//
////#include <pcl/visualization/cloud_viewer.h>
////#include <iostream>//标准C++库中的输入输出类相关头文件。
////#include <pcl/io/io.h>
////#include <pcl/io/pcd_io.h>//pcd 读写类相关的头文件。
////#include <pcl/io/ply_io.h>
////#include <pcl/point_types.h> //PCL中支持的点类型头文件。
////#include<fstream>  
////#include <string>  
////#include <vector> 
////
////using namespace std;
////
////int main()
////{
////	typedef struct tagPOINT_3D
////	{
////		double x;  //mm world coordinate x  
////		double y;  //mm world coordinate y  
////		double z;  //mm world coordinate z  
////		double r;
////	}POINT_WORLD;
////
////
////	// 加载txt数据
////		int number_Txt;
////	FILE* fp_txt;
////	tagPOINT_3D TxtPoint;
////	vector<tagPOINT_3D> m_vTxtPoints;
////	fp_txt = fopen("za.txt", "r");
////	if (fp_txt)
////	{
////		while (fscanf(fp_txt, "%lf %lf %lf", &TxtPoint.x, &TxtPoint.y, &TxtPoint.z) != EOF)
////		{
////			m_vTxtPoints.push_back(TxtPoint);
////		}
////	}
////	else
////		cout << "txt数据加载失败！" << endl;
////	number_Txt = m_vTxtPoints.size();
////	//pcl::PointCloud<pcl::PointXYZ> cloud;
////	//这里使用“PointXYZ”是因为我后面给的点云信息是包含的三维坐标，同时还有点云信息包含的rgb颜色信息的或者还有包含rgba颜色和强度信息。
////	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
////	// Fill in the cloud data  
////	cloud->width = number_Txt;
////	cloud->height = 1;
////	cloud->is_dense = false;
////	cloud->points.resize(cloud->width * cloud->height);
////	for (size_t i = 0; i < cloud->points.size(); ++i)
////	{
////		cloud->points[i].x = m_vTxtPoints[i].x;
////		cloud->points[i].y = m_vTxtPoints[i].y;
////		cloud->points[i].z = m_vTxtPoints[i].z;
////	}
////	pcl::io::savePCDFileASCII("txt2pcd_bunny1.pcd", *cloud);
////	std::cerr << "Saved " << cloud->points.size() << " data points to txt2pcd.pcd." << std::endl;
////
////	//for (size_t i = 0; i < cloud.points.size(); ++i)
////	//  std::cerr << "    " << cloud.points[i].x << " " << cloud.points[i].y << " " << cloud.points[i].z << std::endl;
////
////	//PCL Visualizer
////	// Viewer  
////	pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
////	viewer.addPointCloud(cloud);
////	viewer.setBackgroundColor(0, 0, 0);
////
////	viewer.spin();
////	system("pause");
////	return 0;
////
////}



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

#include <omp.h>

//#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>


#include "CameraModel.h"
#include "Utility.h"


//RGB w x h
const int IMAGE_WIDTH_640 = 640;
const int  IMAGE_HEIGHT_480 = 480;
//Read data outtime
const int  UVC_TIME_OUT = 3000; //ms



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

//void printMatrixInfo(const cv::Mat& matrix, const std::string& name) {
//	std::cout << "Matrix: " << name << "\n";
//	std::cout << "Type: " << matrix.type() << "\n";
//	//std::cout << "Size: " << matrix.size() << "\n";
//	std::cout << "shape: " << "[" << matrix.size().height << ", " << matrix.size().width << "]" << "\n";
//	std::cout << "Channels: " << matrix.channels() << "\n";
//	std::cout << "Depth: " << matrix.depth() << "\n";
//	std::cout << "------------------------------------\n";
//}
//
//void hMirrorTrans(const cv::Mat& src, cv::Mat& dst)
//{
//	dst.create(src.rows, src.cols, src.type());
//
//	int rows = src.rows;
//	int cols = src.cols;
//
//	auto datatype = src.type();
//
//	switch (src.channels())
//	{
//	case 1:   //1通道比如深度图像
//		if (datatype == CV_16UC1)
//		{
//			const ushort* origal;
//			ushort* p;
//			for (int i = 0; i < rows; i++) {
//				origal = src.ptr<ushort>(i);
//				p = dst.ptr<ushort>(i);
//				for (int j = 0; j < cols; j++) {
//					p[j] = origal[cols - 1 - j];
//				}
//			}
//		}
//		else if (datatype == CV_8U)
//		{
//			const uchar* origal;
//			uchar* p;
//			for (int i = 0; i < rows; i++) {
//				origal = src.ptr<uchar>(i);
//				p = dst.ptr<uchar>(i);
//				for (int j = 0; j < cols; j++) {
//					p[j] = origal[cols - 1 - j];
//				}
//			}
//		}
//
//		break;
//	case 3:   //3通道比如彩色图像
//		const cv::Vec3b * origal3;
//		cv::Vec3b* p3;
//		for (int i = 0; i < rows; i++) {
//			origal3 = src.ptr<cv::Vec3b>(i);
//			p3 = dst.ptr<cv::Vec3b>(i);
//			for (int j = 0; j < cols; j++) {
//				p3[j] = origal3[cols - 1 - j];
//			}
//		}
//		break;
//	default:
//		break;
//	}
//
//}
//
//
//
//
//cv::Mat Get3DPoints(const cv::Mat& depth, const cv::Mat& pixels_to_points_map)
//{
//	//cv::Mat points(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_64FC3);
//	cv::Mat points(3, IMAGE_HEIGHT_480 * IMAGE_WIDTH_640, CV_32F);
//
//	// 对depth进行reshape操作，操作后的大小为[1, 640*480]
//	cv::Mat depth_flatten = depth.reshape(1, 1);
//
//	// 将depth_flatten中的每个元素复制到三行
//	cv::Mat depth_flatten_3 = cv::repeat(depth_flatten, 3, 1);
//
//	//转为float
//	cv::Mat depth_flatten_3_float;
//	depth_flatten_3.convertTo(depth_flatten_3_float, CV_32F);
//
//	cv::multiply(depth_flatten_3_float, pixels_to_points_map, points);
//
//	return points;
//}
//
//
//
//cv::Mat GetPixels(const cv::Mat& points, const cv::Mat& camera_matrix, const cv::Mat& depth_map)
//{
//	// 得到深度图每个点对应在rgb图像中的像素
//	cv::Mat pixels = camera_matrix * points;
//
//	// 获取第三行的元素
//	cv::Mat z = pixels.row(2);
//
//	// 创建一个矩阵，每个元素都是对应列的第三行元素的倒数
//	cv::Mat inverse_z;
//	cv::divide(1.0, z, inverse_z);
//	cv::Mat inverse_z_extended = cv::repeat(inverse_z, 3, 1); // [1, 480*640] -> [3, 480*640]
//
//	// 将原始矩阵与倒数矩阵逐元素相乘
//	pixels = pixels.mul(inverse_z_extended);
//
//	// 构建一个行向量，所有元素都是1
//	cv::Mat onesRow = cv::Mat::ones(1, pixels.cols, pixels.type());
//
//	// 获取要检查的行
//	cv::Mat targetRow = pixels.row(2);
//
//	// 使用 cv::compare 检查行是否全部为1
//	cv::Mat comparisonResult;
//	cv::compare(targetRow, onesRow, comparisonResult, cv::CMP_EQ);
//
//	// depth_in_rgb: 存储转换至rgb图像下的深度图
//	cv::Mat depth_in_rgb = cv::Mat::zeros(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_16U);
//
//	cv::Mat map_x = cv::Mat::zeros(depth_map.size(), CV_32FC1);
//	cv::Mat map_y = cv::Mat::zeros(depth_map.size(), CV_32FC1);
//
//
//	//cv::Mat map_x_mask = cv::Mat::zeros(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_16U);
//	// 使用 OpenMP 并行化外层循环
//	//#pragma omp parallel for
//	for (int y = 0; y < IMAGE_HEIGHT_480; ++y)
//	{
//		int column__ = y * IMAGE_WIDTH_640;
//		for (int x = 0; x < IMAGE_WIDTH_640; ++x)
//		{
//			// 计算原始depth(x, y)处的深度值应该在rgb下的位置(x_inrgb, y_inrgb)
//			//int column = y * IMAGE_WIDTH_640 + x;
//			int column = column__ + x;
//			float x_inrgb = pixels.at<float>(0, column);
//			float y_inrgb = pixels.at<float>(1, column);
//
//			if (y_inrgb >= 1 && y_inrgb <= IMAGE_HEIGHT_480 - 1 && x_inrgb >= 1 && x_inrgb <= IMAGE_WIDTH_640 - 1)
//			{
//
//				map_x.at<float>(y_inrgb, x_inrgb) = x;
//				map_y.at<float>(y_inrgb, x_inrgb) = y;
//
//			}
//
//		}
//	}
//
//
//
//	cv::remap(depth_map, depth_in_rgb, map_x, map_y, cv::INTER_LANCZOS4, cv::BORDER_REPLICATE, cv::Scalar());
//	//（1）INTER_NEAREST——最近邻插值
//	//（2）INTER_LINEAR——双线性插值（默认）
//	//（3）INTER_CUBIC——双三样条插值（逾4×4像素邻域内的双三次插值）
//	// (4）INTER_LANCZOS4——lanczos插值（逾8×8像素邻域的Lanczos插值）
//
//	return depth_in_rgb;
//
//}















int main()
{
	try {

		//std::string  depth_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/depth_0.avi";
		//std::string  rgb_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/rgb_0.avi";
		// 存储成图片形式的地址
		std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/DepthImgs1";
		std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/RGBImgs1";


		CameraModel camera_model;


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

		//std::string rgb_file_name = "D:/aaaLab/aaagraduate/SaveVideo/source/1/RGBImgs/rgb_23.png";
		//std::string depth_file_name = "D:/aaaLab/aaagraduate/SaveVideo/source/1/DepthImgs/depth_23.png";


		std::string rgb_file_name = rgb_folder_path + "/rgb_000122.png";  // 83 576 
		std::string depth_file_name = depth_folder_path + "/depth_000122.png";
		std::cout << rgb_file_name << "\n";
		cv::Mat rgb = cv::imread(rgb_file_name);
		cv::Mat depth = cv::imread(depth_file_name, cv::IMREAD_UNCHANGED);
		if (!depth.empty()) {
			std::cout << "Image size: " << depth.size() << std::endl;
			std::cout << "Number of channels: " << depth.channels() << std::endl;
		}

		else {
			std::cerr << "Failed to load image." << std::endl;
		}
		camera_model.printMatrixInfo(depth, "depth");

		//cv::Mat combinedImage;
		//cv::hconcat(rgb, depth, combinedImage);
		//cv::imshow("Combined Images", combinedImage);

		// 获得深度图每个像素点对应的3D空间坐标 (x, y, z)
		//cv::Mat points = Get3DPoints(hImageDepth, pixels_to_points);
		cv::Mat points = camera_model.Get3DPoints(depth, pixels_to_points);


		cv::Mat T_extended = cv::repeat(camera_model.T_depth2rgb, 1, IMAGE_HEIGHT_480 * IMAGE_WIDTH_640); // [3, 1] -> [3, 480*640]
		cv::Mat points_inrgb = camera_model.R_depth2rgb * points + T_extended; // points应该化成(3, 1)的样子，不急，回来改

		//cv::Mat depth_inrgb = GetPixels(points_inrgb, RGBCameraMatrix, hImageDepth);
		cv::Mat depth_inrgb = camera_model.GetPixels(points_inrgb, camera_model.RGBCameraMatrix, depth);

		// 转换至rgb坐标系下，方便构建点云
		cv::Mat points_rgbcoord = camera_model.PixelsCoordTransfer(points_inrgb);
		camera_model.printMatrixInfo(points_rgbcoord, "points_rgbcoord");

		// 中值滤波处理，先测试一下 (Kun: 2024.3.7)
		cv::medianBlur(depth_inrgb, depth_inrgb, 5);

		double max_depth_value;
		// 使用 cv::minMaxLoc 函数获取最大值和位置
		cv::minMaxLoc(depth_inrgb, nullptr, &max_depth_value, nullptr, nullptr);
		std::cout << max_depth_value << '\n';

		// 设定阈值
		float threshold_value = 5000;
		//// 遍历图像，将大于阈值的元素设为0，保持其他元素不变
		//for (int i = 0; i < depth_inrgb.rows; ++i) {
		//	for (int j = 0; j < depth_inrgb.cols; ++j) {
		//		if (depth_inrgb.at<float>(i, j) > threshold_value) {
		//			depth_inrgb.at<float>(i, j) = 0.0f;
		//		}
		//	}
		//}
		// 使用 cv::threshold 函数将大于阈值的像素设为0
		cv::Mat thresholded_depth;
		cv::threshold(depth_inrgb, thresholded_depth, threshold_value, 0, cv::THRESH_TOZERO);



		//float scale_factor = 255.0 / static_cast<float>(max_depth_value);
		float scale_factor = 255.0 / static_cast<float>(threshold_value);

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
		cv::imshow("depth_inrgb_color", depth_inrgb_color);

		// 叠加深度图+rgb图像
		cv::Mat rgb_depth;
		double depthweight = 0.5;
		//cv::addWeighted(depth_inrgb_color, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
		cv::addWeighted(depth_inrgb_color, depthweight, rgb, (1 - depthweight), 0.0, rgb_depth);
		cv::imshow("Mixed", rgb_depth);




		// 显示帧
		//cv::imshow("Camera Feed", frame);
		cv::imshow("Camera Feed", rgb);


		//count++;
		//if (count == 1)
		//	break;


		//// 终止快捷键 ESC
		//if (cv::waitKey(1) == 27)
		//	break;
		//}
		//    std::cout << "Test PCL !!!" << std::endl;

		// ------------------------- PCL Part ------------------------- 
		// 创建RGB图像和对应的空间位置（假设这里是一些随机数据）
		//cv::Mat rgb_image;  // 你的RGB图像
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

		// 填充点云数据，每个点包括位置和颜色信息
		for (int y = 0; y < rgb.rows; ++y) {
			int column__ = y * IMAGE_WIDTH_640;
			for (int x = 0; x < rgb.cols; ++x) {
				int column = column__ + x;
				pcl::PointXYZRGB point;

				// 获取RGB颜色信息
				cv::Vec3b color = rgb.at<cv::Vec3b>(y, x);
				point.r = color[2];
				point.g = color[1];
				point.b = color[0];

				// 假设空间位置是(x, y, 0)，你需要根据你的数据结构进行更改
				//point.x = points_inrgb.at<float>(0, column);
				//point.y = points_inrgb.at<float>(1, column);
				//point.z = points_inrgb.at<float>(2, column);

				point.x = points_rgbcoord.at<cv::Vec3f>(y, x)[0];
				point.y = points_rgbcoord.at<cv::Vec3f>(y, x)[1];
				point.z = points_rgbcoord.at<cv::Vec3f>(y, x)[2];

				//point.z = depth_inrgb.at<float>(y, x);

				//// 假设空间位置是(x, y, 0)，你需要根据你的数据结构进行更改
				//point.x = x;
				//point.y = y;
				//point.z = 0;

				//std::cout << point.x << "  " << point.y << "  " << point.z << "\n";
				if (point.z < threshold_value)
				{
					cloud->push_back(point);
				}
				
			}
		}

		// 可视化点云
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("PointCloud Viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "point_cloud");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point_cloud");
		//cv::waitKey(0);
		// 显示窗口并等待用户交互
		while (!viewer->wasStopped()) {
			viewer->spinOnce(100);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}

		// 创建pcl窗口，显示点云
		//pcl::visualization::CloudViewer viewer("test");
		//viewer.showCloud(point_cloud_ptr);
		//cv::waitKey(0);
		//while (!viewer.wasStopped()) {};

		//// 存入帧数
		//std::ofstream depth_count_file(depth_folder_path + "/frame_num.txt");
		//std::ofstream rgb_count_file(rgb_folder_path + "/frame_num.txt");
		//depth_count_file << count;
		//rgb_count_file << count;
		//depth_count_file.close();
		//rgb_count_file.close();

		//// 关闭数据流
		//streamDepth.destroy();
		////streamColor.destroy();
		//// 关闭设备
		//xtion.close();
		//// 最后关闭OpenNI
		//openni::OpenNI::shutdown();
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










