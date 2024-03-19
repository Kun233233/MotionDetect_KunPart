
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

		double minVal, maxVal;
		// Ѱ����Сֵ�����ֵ
		cv::minMaxLoc(depth_inrgb, &minVal, &maxVal);
		// ���Ա任��floatMat��ֵ���ŵ�0-255�ķ�Χ��
		cv::Mat scaledMat;
		float scale = 255.0 / static_cast<float>((maxVal - minVal));
		float shift = static_cast<float>(-minVal) * scale;

		cv::Mat depth_inrgb_CV8U;
		//cv::convertScaleAbs(depth_inrgb, depth_inrgb_CV8U, scale_factor, offset);
		depth_inrgb.convertTo(depth_inrgb_CV8U, CV_8U, scale, shift);
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










