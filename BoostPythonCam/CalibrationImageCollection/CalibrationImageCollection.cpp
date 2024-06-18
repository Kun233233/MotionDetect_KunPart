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
	// 获取设备信息  
	Array<DeviceInfo> aDeviceList;
	OpenNI::enumerateDevices(&aDeviceList);

	std::cout << "电脑上连接着 " << aDeviceList.getSize() << " 个体感设备." << endl;

	for (int i = 0; i < aDeviceList.getSize(); ++i)
	{
		cout << "设备 " << i << endl;
		const DeviceInfo& rDevInfo = aDeviceList[i];
		cout << "设备名： " << rDevInfo.getName() << endl;
		cout << "设备Id： " << rDevInfo.getUsbProductId() << endl;
		cout << "供应商名： " << rDevInfo.getVendor() << endl;
		cout << "供应商Id: " << rDevInfo.getUsbVendorId() << endl;
		cout << "设备URI: " << rDevInfo.getUri() << endl;

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
	case 1:   //1通道比如深度图像
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
	case 3:   //3通道比如彩色图像
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

int main()
{
	try {

		// 存储成图片形式的地址
		std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/Depth";
		std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/RGB";


		Status rc = STATUS_OK;

		// 初始化OpenNI环境
		OpenNI::initialize();

		showdevice();

		// 声明并打开Device设备。
		Device xtion;
		const char* deviceURL = openni::ANY_DEVICE;  //设备名
		rc = xtion.open(deviceURL);


		// 创建深度数据流
		VideoStream streamDepth;
		rc = streamDepth.create(xtion, SENSOR_IR);
		if (rc == STATUS_OK)
		{
			// 设置深度图像视频模式
			VideoMode mModeDepth;
			// 分辨率大小
			mModeDepth.setResolution(640, 480);
			// 每秒30帧
			mModeDepth.setFps(30);
			// 像素格式
			mModeDepth.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);

			streamDepth.setVideoMode(mModeDepth);

			int nFilterEnable = 0;
			int dataSize = 4;
			//streamDepth.setProperty(XN_STREAM_PROPERTY_HOLE_FILTER, (uint8_t*)&nFilterEnable, dataSize);

			// 打开深度数据流
			rc = streamDepth.start();
			if (rc != STATUS_OK)
			{
				cerr << "无法打开深度数据流：" << OpenNI::getExtendedError() << endl;
				streamDepth.destroy();
			}
		}
		else
		{
			cerr << "无法创建深度数据流：" << OpenNI::getExtendedError() << endl;
		}


		// 打开摄像头，目前先设为1，应该有办法直接读到相机的ID
		Array<DeviceInfo> aDeviceList;
		OpenNI::enumerateDevices(&aDeviceList);
		const DeviceInfo& rDevInfo = aDeviceList[0];
		//cv::VideoCapture cap(rDevInfo.getUsbProductId());
		cv::VideoCapture imgCap(0);

		// 检查摄像头是否成功打开
		if (!imgCap.isOpened()) {
			std::cerr << "Error: Unable to open camera." << std::endl;
			return -1;
		}

		// 设置摄像头参数（可选）
		// 例如，设置分辨率
		imgCap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
		imgCap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);



		// 创建OpenCV图像窗口
		//namedWindow("Depth Image", CV_WINDOW_AUTOSIZE);
		//namedWindow("Color Image", CV_WINDOW_AUTOSIZE);

		// 获得最大深度值
		int iMaxDepth = streamDepth.getMaxPixelValue();

		// 循环读取数据流信息并保存在VideoFrameRef中
		VideoFrameRef  frameDepth;
		//VideoFrameRef  frameColor;

		// 记录循环次数
		int count = 0;

		while (true)
		{
			// 读取数据流
			rc = streamDepth.readFrame(&frameDepth);
			Mat mScaledDepth, hScaledDepth, mImageDepth, hImageDepth;
			if (rc == STATUS_OK)
			{
				// 将深度数据转换成OpenCV格式
				cv::Mat depthtemp(frameDepth.getHeight(), frameDepth.getWidth(), CV_16UC1, (void*)frameDepth.getData()); //CV_16UC1
				mImageDepth = depthtemp;
				// 为了让深度图像显示的更加明显一些，将CV_16UC1 ==> CV_8U格式
				//Mat mScaledDepth, hScaledDepth;

				mImageDepth.convertTo(mScaledDepth, CV_8U, 255.0 / iMaxDepth);

				//水平镜像深度图
				hMirrorTrans(mScaledDepth, hScaledDepth);
				// 显示出深度图像
				//imshow("Depth Image", hScaledDepth);

				// 显示CV_16UC1格式的深度图
				//hMirrorTrans(mImageDepth, hImageDepth);
				//imshow("Origin Depth Image", hImageDepth);
			}



			// 读取帧
			cv::Mat frame;
			imgCap >> frame;

			// 检查帧是否成功读取
			if (frame.empty()) {
				std::cerr << "Error: Failed to capture frame." << std::endl;
				break;
			}

			// 在这里可以对帧进行处理

			// 显示帧
			cv::imshow("Camera Feed", frame);
			cv::imshow("hScaledDepth", hScaledDepth);

			//cv::Mat combinedImage;
			//cv::hconcat(frame, hImageDepth, combinedImage);
			//cv::imshow("combinedImage", combinedImage);

			// 拍照按键 c
			if (cv::waitKey(1) == 99) // c
			{
				// 保存深度图像，格式位 count.png
				std::string depth_file_name = depth_folder_path + "/" + std::to_string(count) + ".png";
				cv::imwrite(depth_file_name, hScaledDepth);

				// 保存rgb图像，格式位 count.png
				std::string rgb_file_name = rgb_folder_path + "/" + std::to_string(count) + ".png";
				cv::imwrite(rgb_file_name, frame);

				count++;
			}



			// 终止快捷键 ESC
			if (cv::waitKey(1) == 27) // ESC
				break;
		}

		// 存入帧数
		std::ofstream depth_count_file(depth_folder_path + "/frame_num.txt");
		std::ofstream rgb_count_file(rgb_folder_path + "/frame_num.txt");
		depth_count_file << count;
		rgb_count_file << count;
		depth_count_file.close();
		rgb_count_file.close();

		// 关闭数据流
		streamDepth.destroy();
		//streamColor.destroy();
		// 关闭设备
		xtion.close();
		// 最后关闭OpenNI
		OpenNI::shutdown();
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


