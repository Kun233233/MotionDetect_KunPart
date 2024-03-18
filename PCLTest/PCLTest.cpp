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
//    // ���� PCL ���ӻ�����
//    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("PointCloud Viewer"));
//    viewer->setBackgroundColor(0.0, 0.0, 0.0);
//
//    // ��������ӵ����ӻ�����
//    viewer->addPointCloud(cloud, "point_cloud");
//
//    // ��ʾ���ӻ�����
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
////#include <iostream>//��׼C++���е�������������ͷ�ļ���
////#include <pcl/io/io.h>
////#include <pcl/io/pcd_io.h>//pcd ��д����ص�ͷ�ļ���
////#include <pcl/io/ply_io.h>
////#include <pcl/point_types.h> //PCL��֧�ֵĵ�����ͷ�ļ���
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
////	// ����txt����
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
////		cout << "txt���ݼ���ʧ�ܣ�" << endl;
////	number_Txt = m_vTxtPoints.size();
////	//pcl::PointCloud<pcl::PointXYZ> cloud;
////	//����ʹ�á�PointXYZ������Ϊ�Һ�����ĵ�����Ϣ�ǰ�������ά���꣬ͬʱ���е�����Ϣ������rgb��ɫ��Ϣ�Ļ��߻��а���rgba��ɫ��ǿ����Ϣ��
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
	// ��ȡ�豸��Ϣ  
	openni::Array<openni::DeviceInfo> aDeviceList;
	openni::OpenNI::enumerateDevices(&aDeviceList);

	std::cout << "������������ " << aDeviceList.getSize() << " ������豸." << endl;

	for (int i = 0; i < aDeviceList.getSize(); ++i)
	{
		cout << "�豸 " << i << endl;
		const openni::DeviceInfo& rDevInfo = aDeviceList[i];
		cout << "�豸���� " << rDevInfo.getName() << endl;
		cout << "�豸Id�� " << rDevInfo.getUsbProductId() << endl;
		cout << "��Ӧ������ " << rDevInfo.getVendor() << endl;
		cout << "��Ӧ��Id: " << rDevInfo.getUsbVendorId() << endl;
		cout << "�豸URI: " << rDevInfo.getUri() << endl;

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
//	case 1:   //1ͨ���������ͼ��
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
//	case 3:   //3ͨ�������ɫͼ��
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
//	// ��depth����reshape������������Ĵ�СΪ[1, 640*480]
//	cv::Mat depth_flatten = depth.reshape(1, 1);
//
//	// ��depth_flatten�е�ÿ��Ԫ�ظ��Ƶ�����
//	cv::Mat depth_flatten_3 = cv::repeat(depth_flatten, 3, 1);
//
//	//תΪfloat
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
//	// �õ����ͼÿ�����Ӧ��rgbͼ���е�����
//	cv::Mat pixels = camera_matrix * points;
//
//	// ��ȡ�����е�Ԫ��
//	cv::Mat z = pixels.row(2);
//
//	// ����һ������ÿ��Ԫ�ض��Ƕ�Ӧ�еĵ�����Ԫ�صĵ���
//	cv::Mat inverse_z;
//	cv::divide(1.0, z, inverse_z);
//	cv::Mat inverse_z_extended = cv::repeat(inverse_z, 3, 1); // [1, 480*640] -> [3, 480*640]
//
//	// ��ԭʼ�����뵹��������Ԫ�����
//	pixels = pixels.mul(inverse_z_extended);
//
//	// ����һ��������������Ԫ�ض���1
//	cv::Mat onesRow = cv::Mat::ones(1, pixels.cols, pixels.type());
//
//	// ��ȡҪ������
//	cv::Mat targetRow = pixels.row(2);
//
//	// ʹ�� cv::compare ������Ƿ�ȫ��Ϊ1
//	cv::Mat comparisonResult;
//	cv::compare(targetRow, onesRow, comparisonResult, cv::CMP_EQ);
//
//	// depth_in_rgb: �洢ת����rgbͼ���µ����ͼ
//	cv::Mat depth_in_rgb = cv::Mat::zeros(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_16U);
//
//	cv::Mat map_x = cv::Mat::zeros(depth_map.size(), CV_32FC1);
//	cv::Mat map_y = cv::Mat::zeros(depth_map.size(), CV_32FC1);
//
//
//	//cv::Mat map_x_mask = cv::Mat::zeros(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_16U);
//	// ʹ�� OpenMP ���л����ѭ��
//	//#pragma omp parallel for
//	for (int y = 0; y < IMAGE_HEIGHT_480; ++y)
//	{
//		int column__ = y * IMAGE_WIDTH_640;
//		for (int x = 0; x < IMAGE_WIDTH_640; ++x)
//		{
//			// ����ԭʼdepth(x, y)�������ֵӦ����rgb�µ�λ��(x_inrgb, y_inrgb)
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
//	//��1��INTER_NEAREST��������ڲ�ֵ
//	//��2��INTER_LINEAR����˫���Բ�ֵ��Ĭ�ϣ�
//	//��3��INTER_CUBIC����˫��������ֵ����4��4���������ڵ�˫���β�ֵ��
//	// (4��INTER_LANCZOS4����lanczos��ֵ����8��8���������Lanczos��ֵ��
//
//	return depth_in_rgb;
//
//}















int main()
{
	try {

		//std::string  depth_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/depth_0.avi";
		//std::string  rgb_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/rgb_0.avi";
		// �洢��ͼƬ��ʽ�ĵ�ַ
		std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/DepthImgs1";
		std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/RGBImgs1";


		CameraModel camera_model;


		// �������ص���Ӧ�ռ������ӳ���ϵ
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
		// ��¼ѭ������
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

		// ������ͼÿ�����ص��Ӧ��3D�ռ����� (x, y, z)
		//cv::Mat points = Get3DPoints(hImageDepth, pixels_to_points);
		cv::Mat points = camera_model.Get3DPoints(depth, pixels_to_points);


		cv::Mat T_extended = cv::repeat(camera_model.T_depth2rgb, 1, IMAGE_HEIGHT_480 * IMAGE_WIDTH_640); // [3, 1] -> [3, 480*640]
		cv::Mat points_inrgb = camera_model.R_depth2rgb * points + T_extended; // pointsӦ�û���(3, 1)�����ӣ�������������

		//cv::Mat depth_inrgb = GetPixels(points_inrgb, RGBCameraMatrix, hImageDepth);
		cv::Mat depth_inrgb = camera_model.GetPixels(points_inrgb, camera_model.RGBCameraMatrix, depth);

		// ת����rgb����ϵ�£����㹹������
		cv::Mat points_rgbcoord = camera_model.PixelsCoordTransfer(points_inrgb);
		camera_model.printMatrixInfo(points_rgbcoord, "points_rgbcoord");

		// ��ֵ�˲������Ȳ���һ�� (Kun: 2024.3.7)
		cv::medianBlur(depth_inrgb, depth_inrgb, 5);

		double max_depth_value;
		// ʹ�� cv::minMaxLoc ������ȡ���ֵ��λ��
		cv::minMaxLoc(depth_inrgb, nullptr, &max_depth_value, nullptr, nullptr);
		std::cout << max_depth_value << '\n';

		// �趨��ֵ
		float threshold_value = 5000;
		//// ����ͼ�񣬽�������ֵ��Ԫ����Ϊ0����������Ԫ�ز���
		//for (int i = 0; i < depth_inrgb.rows; ++i) {
		//	for (int j = 0; j < depth_inrgb.cols; ++j) {
		//		if (depth_inrgb.at<float>(i, j) > threshold_value) {
		//			depth_inrgb.at<float>(i, j) = 0.0f;
		//		}
		//	}
		//}
		// ʹ�� cv::threshold ������������ֵ��������Ϊ0
		cv::Mat thresholded_depth;
		cv::threshold(depth_inrgb, thresholded_depth, threshold_value, 0, cv::THRESH_TOZERO);



		//float scale_factor = 255.0 / static_cast<float>(max_depth_value);
		float scale_factor = 255.0 / static_cast<float>(threshold_value);

		float offset = 0.0;
		cv::Mat depth_inrgb_CV8U;
		cv::convertScaleAbs(depth_inrgb, depth_inrgb_CV8U, scale_factor, offset);
		cv::imshow("depth_inrgb_CV8U", depth_inrgb_CV8U);


		// �����ͼ��һ����0-255��Χ���Ա��� RGB ͼ�����
		cv::Mat depth_inrgb_normalized;
		cv::normalize(depth_inrgb_CV8U, depth_inrgb_normalized, 0, 255, cv::NORM_MINMAX);
		


		// �����ͼת��Ϊ��ͨ�����Ա��� RGB ͼ�����
		cv::Mat depth_inrgb_color;
		cv::applyColorMap(depth_inrgb_normalized, depth_inrgb_color, cv::COLORMAP_JET);
		cv::imshow("depth_inrgb_color", depth_inrgb_color);

		// �������ͼ+rgbͼ��
		cv::Mat rgb_depth;
		double depthweight = 0.5;
		//cv::addWeighted(depth_inrgb_color, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
		cv::addWeighted(depth_inrgb_color, depthweight, rgb, (1 - depthweight), 0.0, rgb_depth);
		cv::imshow("Mixed", rgb_depth);




		// ��ʾ֡
		//cv::imshow("Camera Feed", frame);
		cv::imshow("Camera Feed", rgb);


		//count++;
		//if (count == 1)
		//	break;


		//// ��ֹ��ݼ� ESC
		//if (cv::waitKey(1) == 27)
		//	break;
		//}
		//    std::cout << "Test PCL !!!" << std::endl;

		// ------------------------- PCL Part ------------------------- 
		// ����RGBͼ��Ͷ�Ӧ�Ŀռ�λ�ã�����������һЩ������ݣ�
		//cv::Mat rgb_image;  // ���RGBͼ��
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

		// ���������ݣ�ÿ�������λ�ú���ɫ��Ϣ
		for (int y = 0; y < rgb.rows; ++y) {
			int column__ = y * IMAGE_WIDTH_640;
			for (int x = 0; x < rgb.cols; ++x) {
				int column = column__ + x;
				pcl::PointXYZRGB point;

				// ��ȡRGB��ɫ��Ϣ
				cv::Vec3b color = rgb.at<cv::Vec3b>(y, x);
				point.r = color[2];
				point.g = color[1];
				point.b = color[0];

				// ����ռ�λ����(x, y, 0)������Ҫ����������ݽṹ���и���
				//point.x = points_inrgb.at<float>(0, column);
				//point.y = points_inrgb.at<float>(1, column);
				//point.z = points_inrgb.at<float>(2, column);

				point.x = points_rgbcoord.at<cv::Vec3f>(y, x)[0];
				point.y = points_rgbcoord.at<cv::Vec3f>(y, x)[1];
				point.z = points_rgbcoord.at<cv::Vec3f>(y, x)[2];

				//point.z = depth_inrgb.at<float>(y, x);

				//// ����ռ�λ����(x, y, 0)������Ҫ����������ݽṹ���и���
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

		// ���ӻ�����
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("PointCloud Viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "point_cloud");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point_cloud");
		//cv::waitKey(0);
		// ��ʾ���ڲ��ȴ��û�����
		while (!viewer->wasStopped()) {
			viewer->spinOnce(100);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}

		// ����pcl���ڣ���ʾ����
		//pcl::visualization::CloudViewer viewer("test");
		//viewer.showCloud(point_cloud_ptr);
		//cv::waitKey(0);
		//while (!viewer.wasStopped()) {};

		//// ����֡��
		//std::ofstream depth_count_file(depth_folder_path + "/frame_num.txt");
		//std::ofstream rgb_count_file(rgb_folder_path + "/frame_num.txt");
		//depth_count_file << count;
		//rgb_count_file << count;
		//depth_count_file.close();
		//rgb_count_file.close();

		//// �ر�������
		//streamDepth.destroy();
		////streamColor.destroy();
		//// �ر��豸
		//xtion.close();
		//// ���ر�OpenNI
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










