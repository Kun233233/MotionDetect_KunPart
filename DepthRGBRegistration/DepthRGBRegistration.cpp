
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

void printMatrixInfo(const cv::Mat& matrix, const std::string& name) {
	std::cout << "Matrix: " << name << "\n";
	std::cout << "Type: " << matrix.type() << "\n";
	//std::cout << "Size: " << matrix.size() << "\n";
	std::cout << "shape: " << "[" << matrix.size().height <<", " << matrix.size().width << "]" << "\n";
	std::cout << "Channels: " << matrix.channels() << "\n";
	std::cout << "Depth: " << matrix.depth() << "\n";
	std::cout << "------------------------------------\n";
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




cv::Mat Get3DPoints(const cv::Mat& depth, const cv::Mat& pixels_to_points_map)
{
	//cv::Mat points(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_64FC3);
	cv::Mat points(3, IMAGE_HEIGHT_480 * IMAGE_WIDTH_640, CV_32F);
	// 构建齐次坐标
	//cv::Mat homogeneous_coords_mat(3, 1, CV_32F);
	//homogeneous_coords_mat.at<float>(2, 0) = 1.0;  // 设置齐次坐标的第三个分量为1

	//cv::Mat homogeneous_coords_all(3, IMAGE_HEIGHT_480 * IMAGE_WIDTH_640, CV_32F);
	//homogeneous_coords_all.row(2).setTo(1);

	// 对depth进行reshape操作，操作后的大小为[1, 640*480]
	cv::Mat depth_flatten = depth.reshape(1, 1);

	// 将depth_flatten中的每个元素复制到三行
	cv::Mat depth_flatten_3 = cv::repeat(depth_flatten, 3, 1);

	//转为float
	cv::Mat depth_flatten_3_float;
	depth_flatten_3.convertTo(depth_flatten_3_float, CV_32F);



	//printMatrixInfo(points, "points");
	// 遍历深度图的每个像素
	//for (int y = 0; y < IMAGE_HEIGHT_480; ++y)
	//{
	//	int column__ = y * IMAGE_WIDTH_640;
	//	for (int x = 0; x < IMAGE_WIDTH_640; ++x)
	//	{
	//		int column = column__ + x;
	//		// 获取深度值,从CV16U uint16 转为 float
	//		//std::cout << depth.at<uint16_t>(y, x) << "\n";
	//		float depth_value = static_cast<float>(depth.at<uint16_t>(y, x));
	//		//std::cout << depth_value << "\n";

	//		//// 跳过深度值为零的点
	//		//if (depth_value == 0)
	//		//	continue;

	//		//// 构建齐次坐标
	//		//cv::Point3f homogeneous_coords(x, y, 1.0);
	//		//cv::Mat homogeneous_coords_mat = cv::Mat(homogeneous_coords).clone(); // 使用clone确保深拷贝
	//		//homogeneous_coords_mat.convertTo(homogeneous_coords_mat, CV_32FC1); // 转为32位
	//		//std::cout << cv::Mat(homogeneous_coords) << "\n";

	//		cv::Mat world_coords = depth_value * camera_matrix_inv * homogeneous_coords_mat;

	//		//int column = y * IMAGE_WIDTH_640 + x;
	//		points.at<float>(0, column) = world_coords.at<float>(0, 0);
	//		points.at<float>(1, column) = world_coords.at<float>(1, 0);
	//		points.at<float>(2, column) = world_coords.at<float>(2, 0);
	//	}
	//}

	cv::multiply(depth_flatten_3_float, pixels_to_points_map, points);

	return points;
}


//cv::Mat GetPixels(const cv::Mat& points, const cv::Mat& camera_matrix, const cv::Mat& depth_map)
//{
//	//try{
//	// 得到深度图每个点对应在rgb图像中的像素
//	cv::Mat pixels = camera_matrix * points;
//	//std::cout << pixels.row(2) << "\n";
//
//	// 获取第三行的元素
//	cv::Mat z = pixels.row(2);
//	//std::cout << 1 << "\n";
//
//	// 创建一个矩阵，每个元素都是对应列的第三行元素的倒数
//	cv::Mat inverse_z;
//	cv::divide(1.0, z, inverse_z);
//	//std::cout << 1 << "\n";
//	cv::Mat inverse_z_extended = cv::repeat(inverse_z, 3, 1); // [1, 480*640] -> [3, 480*640]
//	//printMatrixInfo(inverse_z_extended, "inverse_z_extended");
//	//printMatrixInfo(inverse_z, "inverse_z");
//	//printMatrixInfo(pixels, "pixels");
//
//	// 将原始矩阵与倒数矩阵逐元素相乘
//	pixels = pixels.mul(inverse_z_extended);
//	//std::cout << pixels.row(2) << "\n";
//
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
//	// 如果所有元素都为1，返回 true；否则返回 false
//	//std::cout << cv::countNonZero(comparisonResult) - targetRow.cols << "\n";
//
//	
//	//std::cout <<  1 << "\n";
//
//	// depth_in_rgb: 存储转换至rgb图像下的深度图
//	cv::Mat depth_in_rgb(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_16U);
//	//std::cout << 1 << "\n";
//
//	cv::Mat map_x(depth_map.size(), CV_32FC1);
//	cv::Mat map_y(depth_map.size(), CV_32FC1);
//
//	//for (int y = 0; y < IMAGE_HEIGHT_480; ++y)
//	//{
//	//	for (int x = 0; x < IMAGE_WIDTH_640; ++x)
//	//	{
//	//		// 计算原始depth(x, y)处的深度值应该在rgb下的位置(x_inrgb, y_inrgb)
//	//		int column = y * IMAGE_WIDTH_640 + x;
//	//		int x_inrgb = static_cast<int>(pixels.at<float>(0, column)); 
//	//		int y_inrgb = static_cast<int>(pixels.at<float>(1, column));
//
//	//		if (y_inrgb >= 0 && y_inrgb < IMAGE_HEIGHT_480 && x_inrgb >= 0 && x_inrgb < IMAGE_WIDTH_640)
//	//		{
//	//			// 赋值
//	//			//float depth_value = static_cast<float>(depth_map.at<uint16_t>(y, x));
//	//			uint16_t depth_value = depth_map.at<uint16_t>(y, x);
//	//		
//	//			//// 深度值大于阈值的点设为0
//	//			//if (depth_value >= 1500)
//	//			//	depth_value = 0;
//	//		
//	//			//depth_in_rgb.at<float>(y_inrgb, x_inrgb) = depth_value;
//	//			depth_in_rgb.at<uint16_t>(y_inrgb, x_inrgb) = depth_value;
//	//		}
//	//		
//
//	//	}
//	//}
//
//	for (int y = 0; y < IMAGE_HEIGHT_480; ++y)
//	{
//		for (int x = 0; x < IMAGE_WIDTH_640; ++x)
//		{
//			// 计算原始depth(x, y)处的深度值应该在rgb下的位置(x_inrgb, y_inrgb)
//			int column = y * IMAGE_WIDTH_640 + x;
//			float x_inrgb = pixels.at<float>(0, column); 
//			float y_inrgb = pixels.at<float>(1, column);
//
//
//			//// 如果元素大于阈值，将其设为0
//			//if (depth_in_rgb.at<uint>(y, x) > 1500) {
//			//	depth_in_rgb.at<uint>(y, x) = 0;
//			//}
//
//
//			//map_x.at<float>(y, x) = x_inrgb;
//			//map_y.at<float>(y, x) = y_inrgb;
//
//			//map_x.at<float>(y_inrgb, x_inrgb) = x;
//			//map_y.at<float>(y_inrgb, x_inrgb) = y;
//
//			if (y_inrgb >= 0 && y_inrgb <= IMAGE_HEIGHT_480 && x_inrgb >= 0 && x_inrgb <= IMAGE_WIDTH_640)
//			{
//				//map_x.at<float>(y, x) = x_inrgb;
//				//map_y.at<float>(y, x) = y_inrgb;
//				map_x.at<float>(y_inrgb, x_inrgb) = x;
//				map_y.at<float>(y_inrgb, x_inrgb) = y;
//
//			}		
//
//		}
//	}
//
//	cv::remap(depth_map, depth_in_rgb, map_x, map_y, cv::INTER_NEAREST, cv::BORDER_WRAP, cv::Scalar());
//	//（1）INTER_NEAREST——最近邻插值
//	//（2）INTER_LINEAR——双线性插值（默认）
//	//（3）INTER_CUBIC——双三样条插值（逾4×4像素邻域内的双三次插值）
//	// (4）INTER_LANCZOS4——lanczos插值（逾8×8像素邻域的Lanczos插值）
//
//
//
//	return depth_in_rgb;
//
//	
//
//}



cv::Mat GetPixels(const cv::Mat& points, const cv::Mat& camera_matrix, const cv::Mat& depth_map)
{
	// 得到深度图每个点对应在rgb图像中的像素
	cv::Mat pixels = camera_matrix * points;

	// 获取第三行的元素
	cv::Mat z = pixels.row(2);

	// 创建一个矩阵，每个元素都是对应列的第三行元素的倒数
	cv::Mat inverse_z;
	cv::divide(1.0, z, inverse_z);
	cv::Mat inverse_z_extended = cv::repeat(inverse_z, 3, 1); // [1, 480*640] -> [3, 480*640]

	// 将原始矩阵与倒数矩阵逐元素相乘
	pixels = pixels.mul(inverse_z_extended);

	// 构建一个行向量，所有元素都是1
	cv::Mat onesRow = cv::Mat::ones(1, pixels.cols, pixels.type());

	// 获取要检查的行
	cv::Mat targetRow = pixels.row(2);

	// 使用 cv::compare 检查行是否全部为1
	cv::Mat comparisonResult;
	cv::compare(targetRow, onesRow, comparisonResult, cv::CMP_EQ);

	// depth_in_rgb: 存储转换至rgb图像下的深度图
	cv::Mat depth_in_rgb = cv::Mat::zeros(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_16U);

	cv::Mat map_x = cv::Mat::zeros(depth_map.size(), CV_32FC1);
	cv::Mat map_y = cv::Mat::zeros(depth_map.size(), CV_32FC1);


	//cv::Mat map_x_mask = cv::Mat::zeros(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_16U);
	// 使用 OpenMP 并行化外层循环
	//#pragma omp parallel for
	for (int y = 0; y < IMAGE_HEIGHT_480; ++y)
	{
		int column__ = y * IMAGE_WIDTH_640;
		for (int x = 0; x < IMAGE_WIDTH_640; ++x)
		{
			// 计算原始depth(x, y)处的深度值应该在rgb下的位置(x_inrgb, y_inrgb)
			//int column = y * IMAGE_WIDTH_640 + x;
			int column = column__ + x;
			float x_inrgb = pixels.at<float>(0, column);
			float y_inrgb = pixels.at<float>(1, column);

			if (y_inrgb >= 1 && y_inrgb <= IMAGE_HEIGHT_480 - 1 && x_inrgb >= 1 && x_inrgb <= IMAGE_WIDTH_640 - 1)
			{

				map_x.at<float>(y_inrgb, x_inrgb) = x;
				map_y.at<float>(y_inrgb, x_inrgb) = y;

			}

		}
	}


	//for (int y = 1; y < IMAGE_HEIGHT_480 - 1; ++y)
	//{
	//	int column__ = y * IMAGE_WIDTH_640;
	//	for (int x = 1; x < IMAGE_WIDTH_640 - 1; ++x)
	//	{
	//		// 计算相邻元素的平均值，替代零值
	//		map_x.at<float>(y, x) = (map_x.at<float>(y - 1, x) +
	//			map_x.at<float>(y + 1, x) +
	//			map_x.at<float>(y, x - 1) +
	//			map_x.at<float>(y, x + 1)) / 4.0;

	//		map_y.at<float>(y, x) = (map_y.at<float>(y - 1, x) +
	//			map_y.at<float>(y + 1, x) +
	//			map_y.at<float>(y, x - 1) +
	//			map_y.at<float>(y, x + 1)) / 4.0;

	//		

	//	}
	//}

	//// 找到矩阵中为0的元素的索引
	//cv::Mat map_x_mask = (map_x == 0);
	//// 对矩阵进行插值
	//cv::resize(map_x, map_x, map_x.size(), 0, 0, cv::INTER_LINEAR);
	//// 使用插值方法（如 INTER_LINEAR）对0值进行填充
	//map_x.setTo(0, map_x_mask);

	//// 对y同理
	//cv::Mat map_y_mask = (map_y == 0);
	//cv::resize(map_y, map_y, map_y.size(), 0, 0, cv::INTER_LINEAR);
	//map_y.setTo(0, map_y_mask);





	cv::remap(depth_map, depth_in_rgb, map_x, map_y, cv::INTER_LANCZOS4, cv::BORDER_REPLICATE, cv::Scalar());
	//（1）INTER_NEAREST——最近邻插值
	//（2）INTER_LINEAR——双线性插值（默认）
	//（3）INTER_CUBIC——双三样条插值（逾4×4像素邻域内的双三次插值）
	// (4）INTER_LANCZOS4——lanczos插值（逾8×8像素邻域的Lanczos插值）

	return depth_in_rgb;

}















int main()
{
	try {

		std::string  depth_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/depth_0.avi";
		std::string  rgb_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/rgb_0.avi";
		// 存储成图片形式的地址
		std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/src/DepthImgs";
		std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/src/RGBImgs";


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

		// RGB相机内外参
		cv::Mat RGBCameraMatrix = (cv::Mat_<float>(3, 3) <<	597.4286735057399,		0,					356.4896812646391,
																0,						590.3135817242555,	265.6565473501195,
																0,						0,					1				);
		cv::Mat RGBCameraMatrix_inv = RGBCameraMatrix.inv();
		cv::Mat RGBDistCoeffs	= (cv::Mat_<float>(1, 5) << 0.02433779150902952,	0.1806910557398652,		0.01504073394057258,	0.01982505451991676,	-0.4655100996385541);
		cv::Mat RGBRotVec		= (cv::Mat_<float>(1, 3) << 0.1264627521012061,	0.5634184176200842,		1.660489725237417);
		cv::Mat RGBTransVec		= (cv::Mat_<float>(1, 3) << 3.294513374873486,		-4.191418478429084,		20.54231028621967);
		cv::Mat RGBRotationMat;
		cv::Rodrigues(RGBRotVec, RGBRotationMat);

		// 深度相机内外参
		cv::Mat DepthCameraMatrix = (cv::Mat_<float>(3, 3) <<	569.1382523087108,		0,					330.4704844461951,
																0,						564.5139460154893,	250.400178575307,
																0,						0,					1				);
		cv::Mat DepthCameraMatrix_inv	= DepthCameraMatrix.inv();
		cv::Mat DepthDistCoeffs			= (cv::Mat_<float>(1, 5) << -0.1802622269847234,	0.9963006566582099,		-0.001074564774769674,	0.002966307587880594,	-2.140745337976587);
		cv::Mat DepthRotVec				= (cv::Mat_<float>(1, 3) << 0.1313875859188382,	0.62437610031627,		1.648945446919959);
		cv::Mat DepthTransVec			= (cv::Mat_<float>(1, 3) << 6.166359975994443,		-3.53947047998281,		20.74186807903174);
		cv::Mat DepthRotationMat;
		cv::Rodrigues(DepthRotVec, DepthRotationMat);

		// 深度相机转到RGB相机的R和T  P_rgb = R * P_ir + T
		cv::Mat R = RGBRotationMat * DepthRotationMat.inv();
		//std::cout << R << "\n";
		//std::cout << RGBTransVec << "\n";
		//std::cout << DepthTransVec << "\n";
		//std::cout << "RGBTransVec size: " << RGBTransVec.size() << std::endl;
		//std::cout << "RGBTransVec size: " << RGBTransVec.size() << std::endl;
		//std::cout << "DepthTransVec size: " << DepthTransVec.size() << std::endl;
		//std::cout << "R size: " << R.size() << std::endl;
		float squaresize = 12;
		cv::Mat T = (RGBTransVec.t() - R * DepthTransVec.t()) * squaresize;

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
		cv::Mat pixels_to_points = DepthCameraMatrix_inv * homogeneous_coords_all;

		
		int count = 0; // 记录循环次数
		int hole_value = 0; // 定义孔洞深度值
		int kernel_size = 5; // 定义膨胀核的大小
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size)); // 膨胀核

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
				imshow("Depth Image", hScaledDepth);

				// 显示CV_16UC1格式的深度图
				hMirrorTrans(mImageDepth, hImageDepth);
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
			//cv::Mat undistort_frame;
			//cv::undistort(frame, undistort_frame, RGBCameraMatrix, RGBDistCoeffs);
			//cv::imshow("undistort", undistort_frame);

			// 获得深度图每个像素点对应的3D空间坐标 (x, y, z)
			cv::Mat points = Get3DPoints(hImageDepth, pixels_to_points);

			// validation ------------
			//cv::Mat points_x = points.row(0);
			//printMatrixInfo(points_x, "points_x");
			//cv::Mat points_x_map(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_32F);
			////cv::Mat points_x_map = points_x.reshape(1, 480);
			//for (int y = 0; y < IMAGE_HEIGHT_480; ++y)
			//{
			//	for (int x = 0; x < IMAGE_WIDTH_640; ++x)
			//	{
			//		int column = y * IMAGE_WIDTH_640 + x;
			//		points_x_map.at<float>(y, x) = points_x.at<float>(0, column);
			//		//map_y.at<float>(y, x) = y_inrgb;
			//	}
			//}
			////cv::Mat points_x_map = points_x.reshape(1, IMAGE_HEIGHT_480);
			//// 定义变量来存储最大值和最大值的位置
			//double max_x_value;
			//// 使用 cv::minMaxLoc 函数获取最大值和位置
			//cv::minMaxLoc(points_x_map, nullptr, &max_x_value, nullptr, nullptr);
			//float scale_factor_x = 255.0 / static_cast<float>(max_x_value);
			//float offset_x = 0.0;
			//cv::Mat points_x_map_CV8U;
			//cv::convertScaleAbs(points_x_map, points_x_map_CV8U, scale_factor_x);
			//cv::imshow("points_x_map_CV8U", points_x_map_CV8U);


			//printMatrixInfo(R, "R");
			//printMatrixInfo(points, "points");
			//printMatrixInfo(T, "T");
			//printMatrixInfo(RGBRotationMat, "RGBRotationMat");



			cv::Mat T_extended = cv::repeat(T, 1, IMAGE_HEIGHT_480 * IMAGE_WIDTH_640); // [3, 1] -> [3, 480*640]
			cv::Mat points_inrgb = R * points + T_extended; // points应该化成(3, 1)的样子，不急，回来改

			cv::Mat depth_inrgb = GetPixels(points_inrgb, RGBCameraMatrix, hImageDepth);
			//printMatrixInfo(depth_inrgb, "depth_inrgb");
			//cv::imshow("depth_inrgb", depth_inrgb);

			// 孔洞填充
			// 中值滤波
			cv::medianBlur(depth_inrgb, depth_inrgb, 5);
			//// 找到空洞
			//cv::Mat holes = (depth_inrgb == hole_value); 
			//// 使用膨胀操作填充孔洞
			//cv::dilate(holes, holes, kernel);
			//// 使用膨胀后的区域进行插值填充
			//cv::Mat depth_inrgb_filled = depth_inrgb.clone();
			//cv::Mat depth_inrgb_smoothed;
			//cv::medianBlur(depth_inrgb, depth_inrgb_smoothed, 5);
			//depth_inrgb_smoothed.copyTo(depth_inrgb_filled, ~holes);
			//cv::imshow("Depth Map Filled", depth_inrgb_filled);


			//double scale_factor = 255.0 / static_cast<double>(std::numeric_limits<uint16_t>::max());

			double max_depth_value;
			// 使用 cv::minMaxLoc 函数获取最大值和位置
			cv::minMaxLoc(depth_inrgb, nullptr, &max_depth_value, nullptr, nullptr);

			float scale_factor = 255.0 / static_cast<float>(max_depth_value);
			float offset = 0.0;
			cv::Mat depth_inrgb_CV8U;
			cv::Mat depth_inrgb_filled_CV8U;
			cv::convertScaleAbs(depth_inrgb, depth_inrgb_CV8U, scale_factor, offset);
			cv::imshow("depth_inrgb_CV8U", depth_inrgb_CV8U);

			//cv::convertScaleAbs(depth_inrgb_filled, depth_inrgb_filled_CV8U, scale_factor, offset);
			//cv::imshow("Depth Map Filled", depth_inrgb_filled_CV8U);
			//cv::imshow("holes", holes);

			
			
			//// 大于某个深度阈值的设为0
			//for (int i = 0; i < 480; ++i) {
			//	for (int j = 0; j < 640; ++j) {
			//		// 如果元素大于阈值，将其设为0
			//		if (depth_inrgb_CV8U.at<uchar>(i, j) > 128) {
			//			depth_inrgb_CV8U.at<uchar>(i, j) = 0;
			//		}
			//	}
			//}


			// 将深度图归一化到0-255范围，以便与 RGB 图像叠加
			cv::Mat depth_inrgb_normalized;
			cv::normalize(depth_inrgb_CV8U, depth_inrgb_normalized, 0, 255, cv::NORM_MINMAX);


			// 将深度图转换为三通道，以便与 RGB 图像叠加
			cv::Mat depth_inrgb_color;
			cv::applyColorMap(depth_inrgb_normalized, depth_inrgb_color, cv::COLORMAP_JET);

			// 叠加深度图+rgb图像
			cv::Mat rgb_depth;
			double depthweight = 0.5;
			cv::addWeighted(depth_inrgb_color, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
			cv::imshow("Mixed", rgb_depth);




			//std::cout << "Depth Image shape" << hScaledDepth.rows << "  " << hScaledDepth.cols << "\n";

			//if (depth_inrgb.empty()) {
			//	std::cerr << "Error: Depth image is empty." << std::endl;
			//	return -1;
			//}
			
			//cv::Mat scaled_depth_inrgb;
			//cv::normalize(depth_inrgb, scaled_depth_inrgb, 0, 1, cv::NORM_MINMAX);
			//cv::imshow("Float Image", scaled_depth_inrgb * 255.0);

			//Mat depth_inrgb_CV8U = cv::Mat::zeros(depth_inrgb.size(), CV_8U);
			//depth_inrgb_CV8U.convertTo(depth_inrgb, CV_8U, 255.0 / iMaxDepth);
			//printMatrixInfo(depth_inrgb_CV8U, "depth_inrgb_CV8U");
			//cv::imshow("depth_inrgb_CV8U", depth_inrgb);
			//cv::waitKey(100);  // Add this line to wait for a key press before closing the window

			//Mat points_color;
			//points.convertTo(points_color, CV_8U, 255.0 / 10000);
			//cv::imshow("points_color", points_color);





			// 显示帧
			cv::imshow("Camera Feed", frame);



			//// 将深度图归一化到0-255范围，以便与 RGB 图像叠加
			//cv::Mat depthImageNormalized;
			//cv::normalize(hScaledDepth, depthImageNormalized, 0, 255, cv::NORM_MINMAX);

			//// 将深度图转换为三通道，以便与 RGB 图像叠加
			//cv::Mat depthColored;
			//cv::applyColorMap(depth_inrgb_CV8U, depthColored, cv::COLORMAP_JET);

			//// 叠加深度图+rgb图像
			//cv::Mat rgb_depth;
			//double depthweight = 0.5;
			//cv::addWeighted(depthColored, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
			//cv::imshow("Mixed", rgb_depth);
			//std::cout << "Depth Image shape" << hScaledDepth.rows << "  " << hScaledDepth.cols << "\n";





			//// 保存深度图像，格式位depth_(count).png
			//std::string depth_file_name = depth_folder_path + "/depth_" + std::to_string(count) + ".png";
			//cv::imwrite(depth_file_name, hImageDepth);
			//cv::imshow("hImageDepth", hImageDepth);

			//// 保存rgb图像，格式位rgb_(count).png
			//std::string rgb_file_name = rgb_folder_path + "/rgb_" + std::to_string(count) + ".png";
			//cv::imwrite(rgb_file_name, frame);

			count++;








			// 终止快捷键 ESC
			if (waitKey(1) == 27)
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


