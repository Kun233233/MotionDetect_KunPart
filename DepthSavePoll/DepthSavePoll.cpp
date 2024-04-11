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
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>

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

void cap_depth(Status& rc, VideoStream& streamDepth, VideoFrameRef&  frameDepth, cv::Mat& hScaledDepth, cv::Mat& hImageDepth)
{
	rc = streamDepth.readFrame(&frameDepth);
	Mat mScaledDepth, mImageDepth;
	if (rc == STATUS_OK)
	{
		// 将深度数据转换成OpenCV格式
		cv::Mat depthtemp(frameDepth.getHeight(), frameDepth.getWidth(), CV_16UC1, (void*)frameDepth.getData()); //CV_16UC1
		mImageDepth = depthtemp;
		// 为了让深度图像显示的更加明显一些，将CV_16UC1 ==> CV_8U格式
		//Mat mScaledDepth, hScaledDepth;

		mImageDepth.convertTo(mScaledDepth, CV_8U, 255.0 / 1000);

		//水平镜像深度图
		hMirrorTrans(mScaledDepth, hScaledDepth);
		// 显示出深度图像
		imshow("Depth Image", hScaledDepth);

		// 显示CV_16UC1格式的深度图
		hMirrorTrans(mImageDepth, hImageDepth);
		imshow("Origin Depth Image", hImageDepth);
	}

}

void cap_rgb(cv::VideoCapture& imgCap, cv::Mat& frame)
{
	// 读取帧
	//cv::Mat frame;
	imgCap >> frame;

	// 检查帧是否成功读取
	if (frame.empty()) {
		std::cerr << "Error: Failed to capture frame." << std::endl;
		return;
	}

	// 在这里可以对帧进行处理

	// 显示帧
	//cv::imshow("Camera Feed", frame);
}



int main()
{
	try {

		std::string  depth_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/depth_0.avi";
		std::string  rgb_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/rgb_0.avi";
		// 存储成图片形式的地址
		//std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/DepthImgs";
		//std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/RGBImgs";
		//std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/DepthImgs";
		//std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/RGBImgs";
		std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/DepthImgs3";
		std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/RGBImgs3";


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
		rc = streamDepth.create(xtion, SENSOR_DEPTH);
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

		//// 创建彩色图像数据流
		//VideoStream streamColor;
		//rc = streamColor.create(xtion, SENSOR_COLOR);
		//if (rc == STATUS_OK)
		//{
		//	// 同样的设置彩色图像视频模式
		//	VideoMode mModeColor;
		//	mModeColor.setResolution(320, 240);
		//	mModeColor.setFps(30);
		//	mModeColor.setPixelFormat(PIXEL_FORMAT_RGB888);

		//	streamColor.setVideoMode(mModeColor);

		//	// 打开彩色图像数据流
		//	rc = streamColor.start();
		//	if (rc != STATUS_OK)
		//	{
		//		cerr << "无法打开彩色图像数据流：" << OpenNI::getExtendedError() << endl;
		//		streamColor.destroy();
		//	}
		//}
		//else
		//{
		//	cerr << "无法创建彩色图像数据流：" << OpenNI::getExtendedError() << endl;
		//}

		//if (!streamColor.isValid() || !streamDepth.isValid())
		//{
		//	cerr << "彩色或深度数据流不合法" << endl;
		//	OpenNI::shutdown();
		//	return 1;
		//}



		//// 图像模式注册,彩色图与深度图对齐
		//if (xtion.isImageRegistrationModeSupported(
		//	IMAGE_REGISTRATION_DEPTH_TO_COLOR))
		//{
		//	xtion.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
		//}






		//UVC_Swapper uvsSwapper;
		//uvsSwapper.UvcInit();
		//UVCDeviceInfo* deviceInfo = uvsSwapper.getDeviceInfo();
		//printf("UVC device vid= %d, pid = %d.\n", deviceInfo->deviceVid, deviceInfo->devicePid);
		//int fps = 30;
		//if (deviceInfo->devicePid == 0x052b)
		//{
		//	fps = 25;
		//}
		////OB_PIXEL_FORMAT_YUV422 or OB_PIXEL_FORMAT_MJPEG
		//uvsSwapper.UVCStreamStart(IMAGE_WIDTH_640, IMAGE_HEIGHT_480, OB_PIXEL_FORMAT_MJPEG, fps);
		////Data buffer
		//uint8_t* mUvcBuff = new uint8_t[IMAGE_WIDTH_640 * IMAGE_HEIGHT_480 * 2];







		// 打开摄像头
		//cv::VideoCapture cap(1, cv::CAP_V4L2);


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

		//创建视频捕获流
		//cv::VideoWriter depthVideoWriter(depth_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(640, 480), false);
		//cv::VideoWriter depthVideoWriter(depth_video_path, cv::VideoWriter::fourcc('I', '4', '2', '0'), 30, cv::Size(640, 480), false);
		//cv::VideoWriter rgbVideoWriter(rgb_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(640, 480), true);
		//cv::VideoWriter depthVideoWriter(depth_video_path, cv::VideoWriter::fourcc(), 30, cv::Size(640, 480), false);
		//cv::VideoWriter rgbVideoWriter(rgb_video_path, cv::VideoWriter::fourcc(), 30, cv::Size(640, 480), true);

		//if (!depthVideoWriter.isOpened()) {
		//	std::cerr << "Error: Could not open the depth video writer." << std::endl;
		//	return -1;
		//}
		//if (!rgbVideoWriter.isOpened()) {
		//	std::cerr << "Error: Could not open the rgb video writer." << std::endl;
		//	return -1;
		//}









		// 创建OpenCV图像窗口
		namedWindow("Depth Image", CV_WINDOW_AUTOSIZE);
		//namedWindow("Color Image", CV_WINDOW_AUTOSIZE);

		// 获得最大深度值
		int iMaxDepth = streamDepth.getMaxPixelValue();

		// 循环读取数据流信息并保存在VideoFrameRef中
		VideoFrameRef  frameDepth;
		//VideoFrameRef  frameColor;

		// 记录循环次数
		int count = 0;
		int totaltime = 0;

		while (true)
		{
			//auto start_time = std::chrono::high_resolution_clock::now();

			//cv::Mat frame(480, 640, CV_8UC3);
			//std::thread t2(cap_rgb, imgCap, frame);
			//t2.join();

			//auto start_time = std::chrono::high_resolution_clock::now();
			// 读取数据流
			rc = streamDepth.readFrame(&frameDepth);
			//auto end_time = std::chrono::high_resolution_clock::now();
			//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
			//std::cout << "duration: " << duration.count() << '\n';
			//totaltime += duration.count();
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
				imshow("Depth Image", hScaledDepth);

				// 显示CV_16UC1格式的深度图
				hMirrorTrans(mImageDepth, hImageDepth);
				imshow("Origin Depth Image", hImageDepth);
			}


			//uint32_t nSize = 0;
			//uint32_t nImageType = 0;
			//memset(mUvcBuff, 0, IMAGE_WIDTH_640* IMAGE_HEIGHT_480 * 2);
			//int mRet = uvsSwapper.WaitUvcStream(mUvcBuff, nSize, nImageType, UVC_TIME_OUT);
			//if (mRet != CAMERA_STATUS_SUCCESS)
			//{
			//	return mRet;
			//}
			//else
			//{
			//	// 同样的将彩色图像数据转化成OpenCV格式
			//	//const Mat mImageRGB(frameColor.getHeight(), frameColor.getWidth(), CV_8UC3, (void*)frameColor.getData());
			//	const Mat mImageRGB(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_8UC3, mUvcBuff);
			//	// 首先将RGB格式转换为BGR格式
			//	Mat cImageBGR, bImageBGR, hImageBGR;
			//	cvtColor(mImageRGB, cImageBGR, CV_RGB2BGR);

			//	//水平镜像深度图
			//	hMirrorTrans(cImageBGR, hImageBGR);
			//	resize(hImageBGR, hImageBGR, Size(640, 480));
			//	// 然后显示彩色图像
			//	imshow("Color Image", hImageBGR);
			//}





			//rc = streamColor.readFrame(&frameColor);
			//if (rc == STATUS_OK)
			//{
			//	// 同样的将彩色图像数据转化成OpenCV格式
			//	const Mat mImageRGB(frameColor.getHeight(), frameColor.getWidth(), CV_8UC3, (void*)frameColor.getData());
			//	// 首先将RGB格式转换为BGR格式
			//	Mat cImageBGR, bImageBGR, hImageBGR;
			//	cvtColor(mImageRGB, cImageBGR, CV_RGB2BGR);

			//	//水平镜像深度图
			//	hMirrorTrans(cImageBGR, hImageBGR);
			//	resize(hImageBGR, hImageBGR, Size(640, 480));
			//	// 然后显示彩色图像
			//	imshow("Color Image", hImageBGR);
			//}



			//auto start_time = std::chrono::high_resolution_clock::now();

			// 读取帧
			cv::Mat frame;
			imgCap >> frame;

			// 检查帧是否成功读取
			if (frame.empty()) {
				std::cerr << "Error: Failed to capture frame." << std::endl;
				break;
			}

			//// 在这里可以对帧进行处理

			//// 显示帧
			//cv::imshow("Camera Feed", frame);

			//cv::Mat hScaledDepth, hImageDepth, frame;
			//std::thread t1(cap_depth, rc, streamDepth, frameDepth, hScaledDepth, hImageDepth);
			//std::thread t2(cap_rgb, imgCap, frame);
			//t1.join();
			//t2.join();


			//auto end_time = std::chrono::high_resolution_clock::now();
			//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
			//std::cout << "duration: " << duration.count() << '\n';
			//totaltime += duration.count();




			// 将深度图归一化到0-255范围，以便与 RGB 图像叠加
			cv::Mat depthImageNormalized;
			cv::normalize(hScaledDepth, depthImageNormalized, 0, 255, cv::NORM_MINMAX);

			// 将深度图转换为三通道，以便与 RGB 图像叠加
			cv::Mat depthColored;
			//cv::applyColorMap(depthImageNormalized, depthColored, cv::COLORMAP_JET);

			// 叠加深度图+rgb图像
			cv::Mat rgb_depth;
			double depthweight = 0.5;
			//cv::addWeighted(depthColored, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
			//cv::imshow("Mixed", rgb_depth);
			//std::cout << "Depth Image shape" << hScaledDepth.rows << "  " << hScaledDepth.cols << "\n";




			// 高斯滤波深度效果
			cv::Mat depthMid;
			cv::Mat depthFiltered;
			//cv::GaussianBlur(hScaledDepth, depthFiltered, cv::Size(5, 5), 0);
			//cv::imshow("GaussianBlur depth", depthFiltered);

			// 中值滤波
			cv::medianBlur(depthImageNormalized, depthFiltered, 5);
			cv::imshow("medianBlur depth", depthFiltered);

			// 先中值再高斯
			//cv::medianBlur(hScaledDepth, depthMid, 5);
			//cv::GaussianBlur(depthMid, depthFiltered, cv::Size(5, 5), 0);
			//cv::imshow("Gausian+median depth", depthFiltered);

			cv::applyColorMap(hScaledDepth, depthColored, cv::COLORMAP_JET);
			cv::addWeighted(depthColored, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
			cv::imshow("Mixed", rgb_depth);





			//// 捕获图片 保存为视频
			//cv::Mat	depth3Channel[3]{ hImageDepth, hImageDepth, hImageDepth };
			//cv::Mat depth_16UC3, depthColor_16UC3;
			////cv::Mat Red_channel(hImageDepth.size, CV_16UC1);
			////cv::Mat Green_channel([IMAGE_HEIGHT_480, IMAGE_WIDTH_640], CV_16UC1);
			////cv::Mat Blue_channel([IMAGE_HEIGHT_480, IMAGE_WIDTH_640], CV_16UC1);
			////depth3Channel[0] = Red_channel;
			////depth3Channel[1] = Green_channel;
			////depth3Channel[2] = Blue_channel;
			//cv::merge(depth3Channel, 3, depth_16UC3);
			////cv::applyColorMap(depth_16UC3, depthColor_16UC3, cv::COLORMAP_JET);

			//depthVideoWriter.write(hImageDepth);
			//cv::imshow("depth_16UC3", depth_16UC3);
			//rgbVideoWriter.write(frame);

			//depthVideoWriter.write(hScaledDepth);
			//rgbVideoWriter.write(frame);

			// 保存深度图像，格式位depth_(count).png
			std::ostringstream depth_file_name_stream;
			depth_file_name_stream << depth_folder_path << "/depth_" << std::setfill('0') << std::setw(6) << count << ".png";
			std::string depth_file_name = depth_file_name_stream.str();
			//std::string depth_file_name = depth_folder_path + "/depth_" + std::to_string(count) + ".png";
			//cv::imwrite(depth_file_name, hImageDepth, { cv::IMWRITE_PNG_COMPRESSION, 0 }); // 0代表无压缩
			cv::imshow("hImageDepth", hImageDepth);

			//// 读取保存的图像并检查通道数
			//cv::Mat loaded_image = cv::imread(depth_file_name, cv::IMREAD_UNCHANGED);
			//int channels = loaded_image.channels();
			//std::cout << "Number of channels in loaded image: " << channels << std::endl;

			// 保存rgb图像，格式位rgb_(count).png
			std::ostringstream rgb_file_name_stream;
			rgb_file_name_stream << rgb_folder_path << "/rgb_" << std::setfill('0') << std::setw(6) << count << ".png";
			std::string rgb_file_name = rgb_file_name_stream.str();
			//std::string rgb_file_name = rgb_folder_path + "/rgb_" + std::to_string(count) + ".png";
			//cv::imwrite(rgb_file_name, frame);

			count++;


			//std::ostringstream depth_file_name_stream;
			//depth_file_name_stream << depth_folder_path << "/depth_" << std::setfill('0') << std::setw(6) << count << ".png";
			//std::string depth_file_name = depth_file_name_stream.str();







			// 终止快捷键 ESC
			if (waitKey(1) == 27)
				break;

			if (count == 2000)
			{
				break;
			}
		}
		std::cout << "avg time: " << float(totaltime) / float(count) << '\n';

		// 存入帧数
		//std::ofstream depth_count_file(depth_folder_path + "/frame_num.txt");
		//std::ofstream rgb_count_file(rgb_folder_path + "/frame_num.txt");
		//depth_count_file << count;
		//rgb_count_file << count;
		//depth_count_file.close();
		//rgb_count_file.close();

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


