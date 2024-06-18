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

int main()
{
	try {

		// �洢��ͼƬ��ʽ�ĵ�ַ
		std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/Depth";
		std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/RGB";


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
		rc = streamDepth.create(xtion, SENSOR_IR);
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


		// ������ͷ��Ŀǰ����Ϊ1��Ӧ���а취ֱ�Ӷ��������ID
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



		// ����OpenCVͼ�񴰿�
		//namedWindow("Depth Image", CV_WINDOW_AUTOSIZE);
		//namedWindow("Color Image", CV_WINDOW_AUTOSIZE);

		// ���������ֵ
		int iMaxDepth = streamDepth.getMaxPixelValue();

		// ѭ����ȡ��������Ϣ��������VideoFrameRef��
		VideoFrameRef  frameDepth;
		//VideoFrameRef  frameColor;

		// ��¼ѭ������
		int count = 0;

		while (true)
		{
			// ��ȡ������
			rc = streamDepth.readFrame(&frameDepth);
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
				//imshow("Depth Image", hScaledDepth);

				// ��ʾCV_16UC1��ʽ�����ͼ
				//hMirrorTrans(mImageDepth, hImageDepth);
				//imshow("Origin Depth Image", hImageDepth);
			}



			// ��ȡ֡
			cv::Mat frame;
			imgCap >> frame;

			// ���֡�Ƿ�ɹ���ȡ
			if (frame.empty()) {
				std::cerr << "Error: Failed to capture frame." << std::endl;
				break;
			}

			// ��������Զ�֡���д���

			// ��ʾ֡
			cv::imshow("Camera Feed", frame);
			cv::imshow("hScaledDepth", hScaledDepth);

			//cv::Mat combinedImage;
			//cv::hconcat(frame, hImageDepth, combinedImage);
			//cv::imshow("combinedImage", combinedImage);

			// ���հ��� c
			if (cv::waitKey(1) == 99) // c
			{
				// �������ͼ�񣬸�ʽλ count.png
				std::string depth_file_name = depth_folder_path + "/" + std::to_string(count) + ".png";
				cv::imwrite(depth_file_name, hScaledDepth);

				// ����rgbͼ�񣬸�ʽλ count.png
				std::string rgb_file_name = rgb_folder_path + "/" + std::to_string(count) + ".png";
				cv::imwrite(rgb_file_name, frame);

				count++;
			}



			// ��ֹ��ݼ� ESC
			if (cv::waitKey(1) == 27) // ESC
				break;
		}

		// ����֡��
		std::ofstream depth_count_file(depth_folder_path + "/frame_num.txt");
		std::ofstream rgb_count_file(rgb_folder_path + "/frame_num.txt");
		depth_count_file << count;
		rgb_count_file << count;
		depth_count_file.close();
		rgb_count_file.close();

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


