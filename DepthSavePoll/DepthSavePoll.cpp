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

void cap_depth(Status& rc, VideoStream& streamDepth, VideoFrameRef&  frameDepth, cv::Mat& hScaledDepth, cv::Mat& hImageDepth)
{
	rc = streamDepth.readFrame(&frameDepth);
	Mat mScaledDepth, mImageDepth;
	if (rc == STATUS_OK)
	{
		// ���������ת����OpenCV��ʽ
		cv::Mat depthtemp(frameDepth.getHeight(), frameDepth.getWidth(), CV_16UC1, (void*)frameDepth.getData()); //CV_16UC1
		mImageDepth = depthtemp;
		// Ϊ�������ͼ����ʾ�ĸ�������һЩ����CV_16UC1 ==> CV_8U��ʽ
		//Mat mScaledDepth, hScaledDepth;

		mImageDepth.convertTo(mScaledDepth, CV_8U, 255.0 / 1000);

		//ˮƽ�������ͼ
		hMirrorTrans(mScaledDepth, hScaledDepth);
		// ��ʾ�����ͼ��
		imshow("Depth Image", hScaledDepth);

		// ��ʾCV_16UC1��ʽ�����ͼ
		hMirrorTrans(mImageDepth, hImageDepth);
		imshow("Origin Depth Image", hImageDepth);
	}

}

void cap_rgb(cv::VideoCapture& imgCap, cv::Mat& frame)
{
	// ��ȡ֡
	//cv::Mat frame;
	imgCap >> frame;

	// ���֡�Ƿ�ɹ���ȡ
	if (frame.empty()) {
		std::cerr << "Error: Failed to capture frame." << std::endl;
		return;
	}

	// ��������Զ�֡���д���

	// ��ʾ֡
	//cv::imshow("Camera Feed", frame);
}



int main()
{
	try {

		std::string  depth_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/depth_0.avi";
		std::string  rgb_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/rgb_0.avi";
		// �洢��ͼƬ��ʽ�ĵ�ַ
		//std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/DepthImgs";
		//std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/RGBImgs";
		//std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/DepthImgs";
		//std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240314/RGBImgs";
		std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/DepthImgs3";
		std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240326/RGBImgs3";


		Status rc = STATUS_OK;

		// ��ʼ��OpenNI����
		OpenNI::initialize();

		showdevice();

		// ��������Device�豸��
		Device xtion;
		const char* deviceURL = openni::ANY_DEVICE;  //�豸��
		rc = xtion.open(deviceURL);


		// �������������
		VideoStream streamDepth;
		rc = streamDepth.create(xtion, SENSOR_DEPTH);
		if (rc == STATUS_OK)
		{
			// �������ͼ����Ƶģʽ
			VideoMode mModeDepth;
			// �ֱ��ʴ�С
			mModeDepth.setResolution(640, 480);
			// ÿ��30֡
			mModeDepth.setFps(30);
			// ���ظ�ʽ
			mModeDepth.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);

			streamDepth.setVideoMode(mModeDepth);

			int nFilterEnable = 0;
			int dataSize = 4;
			//streamDepth.setProperty(XN_STREAM_PROPERTY_HOLE_FILTER, (uint8_t*)&nFilterEnable, dataSize);

			// �����������
			rc = streamDepth.start();
			if (rc != STATUS_OK)
			{
				cerr << "�޷��������������" << OpenNI::getExtendedError() << endl;
				streamDepth.destroy();
			}
		}
		else
		{
			cerr << "�޷����������������" << OpenNI::getExtendedError() << endl;
		}

		//// ������ɫͼ��������
		//VideoStream streamColor;
		//rc = streamColor.create(xtion, SENSOR_COLOR);
		//if (rc == STATUS_OK)
		//{
		//	// ͬ�������ò�ɫͼ����Ƶģʽ
		//	VideoMode mModeColor;
		//	mModeColor.setResolution(320, 240);
		//	mModeColor.setFps(30);
		//	mModeColor.setPixelFormat(PIXEL_FORMAT_RGB888);

		//	streamColor.setVideoMode(mModeColor);

		//	// �򿪲�ɫͼ��������
		//	rc = streamColor.start();
		//	if (rc != STATUS_OK)
		//	{
		//		cerr << "�޷��򿪲�ɫͼ����������" << OpenNI::getExtendedError() << endl;
		//		streamColor.destroy();
		//	}
		//}
		//else
		//{
		//	cerr << "�޷�������ɫͼ����������" << OpenNI::getExtendedError() << endl;
		//}

		//if (!streamColor.isValid() || !streamDepth.isValid())
		//{
		//	cerr << "��ɫ��������������Ϸ�" << endl;
		//	OpenNI::shutdown();
		//	return 1;
		//}



		//// ͼ��ģʽע��,��ɫͼ�����ͼ����
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







		// ������ͷ
		//cv::VideoCapture cap(1, cv::CAP_V4L2);


		Array<DeviceInfo> aDeviceList;
		OpenNI::enumerateDevices(&aDeviceList);
		const DeviceInfo& rDevInfo = aDeviceList[0];
		//cv::VideoCapture cap(rDevInfo.getUsbProductId());
		cv::VideoCapture imgCap(0);

		// �������ͷ�Ƿ�ɹ���
		if (!imgCap.isOpened()) {
			std::cerr << "Error: Unable to open camera." << std::endl;
			return -1;
		}

		// ��������ͷ��������ѡ��
		// ���磬���÷ֱ���
		imgCap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
		imgCap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

		//������Ƶ������
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









		// ����OpenCVͼ�񴰿�
		namedWindow("Depth Image", CV_WINDOW_AUTOSIZE);
		//namedWindow("Color Image", CV_WINDOW_AUTOSIZE);

		// ���������ֵ
		int iMaxDepth = streamDepth.getMaxPixelValue();

		// ѭ����ȡ��������Ϣ��������VideoFrameRef��
		VideoFrameRef  frameDepth;
		//VideoFrameRef  frameColor;

		// ��¼ѭ������
		int count = 0;
		int totaltime = 0;

		while (true)
		{
			//auto start_time = std::chrono::high_resolution_clock::now();

			//cv::Mat frame(480, 640, CV_8UC3);
			//std::thread t2(cap_rgb, imgCap, frame);
			//t2.join();

			//auto start_time = std::chrono::high_resolution_clock::now();
			// ��ȡ������
			rc = streamDepth.readFrame(&frameDepth);
			//auto end_time = std::chrono::high_resolution_clock::now();
			//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
			//std::cout << "duration: " << duration.count() << '\n';
			//totaltime += duration.count();
			Mat mScaledDepth, hScaledDepth, mImageDepth, hImageDepth;
			if (rc == STATUS_OK)
			{
				// ���������ת����OpenCV��ʽ
				cv::Mat depthtemp(frameDepth.getHeight(), frameDepth.getWidth(), CV_16UC1, (void*)frameDepth.getData()); //CV_16UC1
				mImageDepth = depthtemp;
				// Ϊ�������ͼ����ʾ�ĸ�������һЩ����CV_16UC1 ==> CV_8U��ʽ
				//Mat mScaledDepth, hScaledDepth;

				mImageDepth.convertTo(mScaledDepth, CV_8U, 255.0 / iMaxDepth);

				

				//ˮƽ�������ͼ
				hMirrorTrans(mScaledDepth, hScaledDepth);
				// ��ʾ�����ͼ��
				imshow("Depth Image", hScaledDepth);

				// ��ʾCV_16UC1��ʽ�����ͼ
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
			//	// ͬ���Ľ���ɫͼ������ת����OpenCV��ʽ
			//	//const Mat mImageRGB(frameColor.getHeight(), frameColor.getWidth(), CV_8UC3, (void*)frameColor.getData());
			//	const Mat mImageRGB(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_8UC3, mUvcBuff);
			//	// ���Ƚ�RGB��ʽת��ΪBGR��ʽ
			//	Mat cImageBGR, bImageBGR, hImageBGR;
			//	cvtColor(mImageRGB, cImageBGR, CV_RGB2BGR);

			//	//ˮƽ�������ͼ
			//	hMirrorTrans(cImageBGR, hImageBGR);
			//	resize(hImageBGR, hImageBGR, Size(640, 480));
			//	// Ȼ����ʾ��ɫͼ��
			//	imshow("Color Image", hImageBGR);
			//}





			//rc = streamColor.readFrame(&frameColor);
			//if (rc == STATUS_OK)
			//{
			//	// ͬ���Ľ���ɫͼ������ת����OpenCV��ʽ
			//	const Mat mImageRGB(frameColor.getHeight(), frameColor.getWidth(), CV_8UC3, (void*)frameColor.getData());
			//	// ���Ƚ�RGB��ʽת��ΪBGR��ʽ
			//	Mat cImageBGR, bImageBGR, hImageBGR;
			//	cvtColor(mImageRGB, cImageBGR, CV_RGB2BGR);

			//	//ˮƽ�������ͼ
			//	hMirrorTrans(cImageBGR, hImageBGR);
			//	resize(hImageBGR, hImageBGR, Size(640, 480));
			//	// Ȼ����ʾ��ɫͼ��
			//	imshow("Color Image", hImageBGR);
			//}



			//auto start_time = std::chrono::high_resolution_clock::now();

			// ��ȡ֡
			cv::Mat frame;
			imgCap >> frame;

			// ���֡�Ƿ�ɹ���ȡ
			if (frame.empty()) {
				std::cerr << "Error: Failed to capture frame." << std::endl;
				break;
			}

			//// ��������Զ�֡���д���

			//// ��ʾ֡
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




			// �����ͼ��һ����0-255��Χ���Ա��� RGB ͼ�����
			cv::Mat depthImageNormalized;
			cv::normalize(hScaledDepth, depthImageNormalized, 0, 255, cv::NORM_MINMAX);

			// �����ͼת��Ϊ��ͨ�����Ա��� RGB ͼ�����
			cv::Mat depthColored;
			//cv::applyColorMap(depthImageNormalized, depthColored, cv::COLORMAP_JET);

			// �������ͼ+rgbͼ��
			cv::Mat rgb_depth;
			double depthweight = 0.5;
			//cv::addWeighted(depthColored, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
			//cv::imshow("Mixed", rgb_depth);
			//std::cout << "Depth Image shape" << hScaledDepth.rows << "  " << hScaledDepth.cols << "\n";




			// ��˹�˲����Ч��
			cv::Mat depthMid;
			cv::Mat depthFiltered;
			//cv::GaussianBlur(hScaledDepth, depthFiltered, cv::Size(5, 5), 0);
			//cv::imshow("GaussianBlur depth", depthFiltered);

			// ��ֵ�˲�
			cv::medianBlur(depthImageNormalized, depthFiltered, 5);
			cv::imshow("medianBlur depth", depthFiltered);

			// ����ֵ�ٸ�˹
			//cv::medianBlur(hScaledDepth, depthMid, 5);
			//cv::GaussianBlur(depthMid, depthFiltered, cv::Size(5, 5), 0);
			//cv::imshow("Gausian+median depth", depthFiltered);

			cv::applyColorMap(hScaledDepth, depthColored, cv::COLORMAP_JET);
			cv::addWeighted(depthColored, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
			cv::imshow("Mixed", rgb_depth);





			//// ����ͼƬ ����Ϊ��Ƶ
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

			// �������ͼ�񣬸�ʽλdepth_(count).png
			std::ostringstream depth_file_name_stream;
			depth_file_name_stream << depth_folder_path << "/depth_" << std::setfill('0') << std::setw(6) << count << ".png";
			std::string depth_file_name = depth_file_name_stream.str();
			//std::string depth_file_name = depth_folder_path + "/depth_" + std::to_string(count) + ".png";
			//cv::imwrite(depth_file_name, hImageDepth, { cv::IMWRITE_PNG_COMPRESSION, 0 }); // 0������ѹ��
			cv::imshow("hImageDepth", hImageDepth);

			//// ��ȡ�����ͼ�񲢼��ͨ����
			//cv::Mat loaded_image = cv::imread(depth_file_name, cv::IMREAD_UNCHANGED);
			//int channels = loaded_image.channels();
			//std::cout << "Number of channels in loaded image: " << channels << std::endl;

			// ����rgbͼ�񣬸�ʽλrgb_(count).png
			std::ostringstream rgb_file_name_stream;
			rgb_file_name_stream << rgb_folder_path << "/rgb_" << std::setfill('0') << std::setw(6) << count << ".png";
			std::string rgb_file_name = rgb_file_name_stream.str();
			//std::string rgb_file_name = rgb_folder_path + "/rgb_" + std::to_string(count) + ".png";
			//cv::imwrite(rgb_file_name, frame);

			count++;


			//std::ostringstream depth_file_name_stream;
			//depth_file_name_stream << depth_folder_path << "/depth_" << std::setfill('0') << std::setw(6) << count << ".png";
			//std::string depth_file_name = depth_file_name_stream.str();







			// ��ֹ��ݼ� ESC
			if (waitKey(1) == 27)
				break;

			if (count == 2000)
			{
				break;
			}
		}
		std::cout << "avg time: " << float(totaltime) / float(count) << '\n';

		// ����֡��
		//std::ofstream depth_count_file(depth_folder_path + "/frame_num.txt");
		//std::ofstream rgb_count_file(rgb_folder_path + "/frame_num.txt");
		//depth_count_file << count;
		//rgb_count_file << count;
		//depth_count_file.close();
		//rgb_count_file.close();

		// �ر�������
		streamDepth.destroy();
		//streamColor.destroy();
		// �ر��豸
		xtion.close();
		// ���ر�OpenNI
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


