#include <iostream>
#include <OpenNI.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>

#include "OniSampleUtilities.h"
#include "UVC_Swapper.h"
#include "UVCSwapper.h"
#include "OBTypes.h"
#include "ObCommon.h"
//#include "OniSampleUtilities.h"

#include <fstream>

using namespace std;
using namespace openni;
using namespace cv;

//RGB w x h
const int IMAGE_WIDTH_640 = 640;
const int  IMAGE_HEIGHT_480 = 480;
//Read data outtime
const int  UVC_TIME_OUT = 3000; //ms

void showdevice() {
	// ��ȡ�豸��Ϣ  
	Array<DeviceInfo> aDeviceList;
	OpenNI::enumerateDevices(&aDeviceList);

	std::cout << "������������ " << aDeviceList.getSize() << " ������豸." << endl;

	for (int i = 0; i < aDeviceList.getSize(); ++i)
	{
		cout << "�豸 " << i << endl;
		const DeviceInfo& rDevInfo = aDeviceList[i];
		cout << "�豸���� " << rDevInfo.getName() << endl;
		cout << "�豸Id�� " << rDevInfo.getUsbProductId() << endl;
		cout << "��Ӧ������ " << rDevInfo.getVendor() << endl;
		cout << "��Ӧ��Id: " << rDevInfo.getUsbVendorId() << endl;
		cout << "�豸URI: " << rDevInfo.getUri() << endl;

	}
}

void hMirrorTrans(const Mat& src, Mat& dst)
{
	dst.create(src.rows, src.cols, src.type());

	int rows = src.rows;
	int cols = src.cols;

	auto datatype = src.type();

	switch (src.channels())
	{
	case 1:   //1ͨ���������ͼ��
		if (datatype == CV_16UC1)
		{
			const ushort* origal;
			ushort* p;
			for (int i = 0; i < rows; i++) {
				origal = src.ptr<ushort>(i);
				p = dst.ptr<ushort>(i);
				for (int j = 0; j < cols; j++) {
					p[j] = origal[cols - 1 - j];
				}
			}
		}
		else if (datatype == CV_8U) 
		{
			const uchar* origal;
			uchar* p;
			for (int i = 0; i < rows; i++) {
				origal = src.ptr<uchar>(i);
				p = dst.ptr<uchar>(i);
				for (int j = 0; j < cols; j++) {
					p[j] = origal[cols - 1 - j];
				}
			}
		}
			
		break;
	case 3:   //3ͨ�������ɫͼ��
		const Vec3b * origal3;
		Vec3b* p3;
		for (int i = 0; i < rows; i++) {
			origal3 = src.ptr<Vec3b>(i);
			p3 = dst.ptr<Vec3b>(i);
			for (int j = 0; j < cols; j++) {
				p3[j] = origal3[cols - 1 - j];
			}
		}
		break;
	default:
		break;
	}

}

int getImagePathList(std::string folder, std::vector<cv::String>& imagePathList)
{
	//search all the image in a folder
	cv::glob(folder, imagePathList);
	return 0;
}


int main()
{
	try {


	std::string  depth_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/depth_0.avi";
	std::string  rgb_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/rgb_0.avi";
	// �洢��ͼƬ��ʽ�ĵ�ַ
	std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/src/DepthImgs";
	std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/src/RGBImgs";

	//std::vector<cv::String> rgbPathList;
	//getImagePathList(rgb_folder_path, rgbPathList);


	// ���ļ�
	std::ifstream rgb_count_file(rgb_folder_path + "/frame_num.txt");

	// ����ļ��Ƿ�ɹ���
	if (!rgb_count_file.is_open()) {
		std::cerr << "Unable to open the file." << std::endl;
		return 1;
	}

	// ��ȡ�ļ��е�����
	int countmax;
	rgb_count_file >> countmax;
	// �ر��ļ�
	rgb_count_file.close();

	// ��¼ѭ������
	//int countmax = 300;
	double fps = 30.0;
	int wait_time = static_cast<int>(1 / fps * 1000.0);

	for(int i = 0; i < countmax; i++)
	{
		cv::namedWindow("Combined Images", cv::WINDOW_NORMAL);
		cv::resizeWindow("Combined Images", 640*2, 480);  // ���ô��ڴ�С
		//std::cout << rgbPathList[i] << "\n";
		std::string rgb_file_name = rgb_folder_path + "/rgb_" + std::to_string(i) + ".png";
		std::string depth_file_name = depth_folder_path + "/depth_" + std::to_string(i) + ".png";
		std::cout << rgb_file_name << "\n";
		cv::Mat rgb = cv::imread(rgb_file_name);
		cv::Mat depth = cv::imread(depth_file_name);
		//cv::imshow("rgb", rgb);
		//cv::imshow("depth", depth);

		cv::Mat combinedImage;
		cv::hconcat(rgb, depth, combinedImage);
		cv::imshow("Combined Images", combinedImage);




		cv::waitKey(wait_time);


	
		// ��ֹ��ݼ� ESC
		if (waitKey(1) == 27)
			break;
	}

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




//#include <opencv2/opencv.hpp>
//#include <opencv2/core/cuda.hpp>
//int main() {
//    //����Ƿ�֧��CUDA
//    if (cv::cuda::getCudaEnabledDeviceCount()) {
//        std::cout << "��⵽֧��CUDA���豸����" << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
//    }
//    else {
//        std::cout << "δ��⵽֧��CUDA���豸" << std::endl;
//    }
//    return 0;
//}



