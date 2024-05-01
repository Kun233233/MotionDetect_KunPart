#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB

//# include <boost/python.hpp>
//#include <boost/python/module.hpp>
//#include <boost/python/def.hpp>
//#include <boost/python/numpy.hpp>
//# include <opencv2/opencv.hpp>
//#include <conversion.h>
#include "BoostPythonCam.h"


namespace np = boost::python::numpy;
using json = nlohmann::json;



RGBDCamera::RGBDCamera() 
	:	camera_model(), 
		utility(), 
		m_cvt()
{
	// 计算像素到对应空间点坐标映射关系
	cv::Mat homogeneous_coords_all(3, camera_model.height * camera_model.width, CV_32F);
	homogeneous_coords_all.row(2).setTo(1);
	for (int y = 0; y < camera_model.height; ++y)
	{
		int column__ = y * camera_model.width;
		for (int x = 0; x < camera_model.width; ++x)
		{
			int column = column__ + x;
			homogeneous_coords_all.at<float>(0, column) = static_cast<float>(x);
			homogeneous_coords_all.at<float>(1, column) = static_cast<float>(y);
		}
	}
	camera_model.depth_pixels_to_points = camera_model.DepthCameraMatrix_inv * homogeneous_coords_all;

	// 记录原始depth中为0的点在rgb坐标系下对应的坐标，便于剔除异常数据
	cv::Mat nodepth_point = (cv::Mat_<float>(3, 1) << 0.0f, 0.0f, 0.0f);
	m_nodepth_point_inrgb = camera_model.R_depth2rgb * nodepth_point + camera_model.T_depth2rgb;

	m_feature_points_now.create(3, 6, CV_32F);
	m_motion_now.create(6, 6, CV_32F);

	m_rgb.create(camera_model.height, camera_model.width, CV_8UC3);
	m_depth.create(camera_model.height, camera_model.width, CV_16UC1);
}

RGBDCamera::RGBDCamera(const std::string& param_path, bool has_device, int rgb_cam_id) 
	:	camera_model(param_path), 
		utility(), 
		m_cvt()
{
	// 计算像素到对应空间点坐标映射关系
	cv::Mat homogeneous_coords_all(3, camera_model.height * camera_model.width, CV_32F);
	homogeneous_coords_all.row(2).setTo(1);
	for (int y = 0; y < camera_model.height; ++y)
	{
		int column__ = y * camera_model.width;
		for (int x = 0; x < camera_model.width; ++x)
		{
			int column = column__ + x;
			homogeneous_coords_all.at<float>(0, column) = x;
			homogeneous_coords_all.at<float>(1, column) = y;
		}
	}
	camera_model.depth_pixels_to_points = camera_model.DepthCameraMatrix_inv * homogeneous_coords_all;

	// 记录原始depth中为0的点在rgb坐标系下对应的坐标，便于剔除异常数据
	cv::Mat nodepth_point = (cv::Mat_<float>(3, 1) << 0.0f, 0.0f, 0.0f);
	m_nodepth_point_inrgb = camera_model.R_depth2rgb * nodepth_point + camera_model.T_depth2rgb;

	m_feature_points_now.create(3, 6, CV_32F);
	m_motion_now.create(6, 6, CV_32F);

	m_rgb.create(camera_model.height, camera_model.width, CV_8UC3);
	m_depth.create(camera_model.height, camera_model.width, CV_16UC1);

	// has_device == true: 对相机采集相关函数初始化
	m_rgb_cam_id = rgb_cam_id;
	m_has_device = has_device;
	if (m_has_device == true)
	{
		m_rc = openni::STATUS_OK;

		// 初始化OpenNI环境
		openni::OpenNI::initialize();

		showdevice();

		// 声明并打开Device设备。
		//openni::Device xtion;
		const char* deviceURL = openni::ANY_DEVICE;  //设备名
		m_rc = m_xtion.open(deviceURL);


		// 创建深度数据流
		m_rc = m_streamDepth.create(m_xtion, openni::SENSOR_DEPTH);
		if (m_rc == openni::STATUS_OK)
		{
			// 设置深度图像视频模式
			openni::VideoMode mModeDepth;
			// 分辨率大小
			mModeDepth.setResolution(640, 480);
			// 每秒30帧
			mModeDepth.setFps(30);
			// 像素格式
			mModeDepth.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);

			m_streamDepth.setVideoMode(mModeDepth);

			int nFilterEnable = 0;
			int dataSize = 4;
			//streamDepth.setProperty(XN_STREAM_PROPERTY_HOLE_FILTER, (uint8_t*)&nFilterEnable, dataSize);

			// 打开深度数据流
			m_rc = m_streamDepth.start();
			if (m_rc != openni::STATUS_OK)
			{
				std::cout << "无法打开深度数据流：" << openni::OpenNI::getExtendedError() << "\n";
				m_streamDepth.destroy();
			}
		}
		else
		{
			std::cout << "无法创建深度数据流：" << openni::OpenNI::getExtendedError() << "\n";
		}




		// 创建RGB相机采集流
		m_imgCap.open(m_rgb_cam_id);

		// 检查摄像头是否成功打开
		if (!m_imgCap.isOpened()) {
			std::cerr << "Error: Unable to open camera." << std::endl;
			return;
		}

		// 设置摄像头参数（可选）
		// 例如，设置分辨率
		m_imgCap.set(cv::CAP_PROP_FRAME_WIDTH, camera_model.width);
		m_imgCap.set(cv::CAP_PROP_FRAME_HEIGHT, camera_model.height);

	}
	
}

RGBDCamera::~RGBDCamera()
{
	if (m_has_device)
	{
		// 关闭数据流
		m_streamDepth.destroy();
		//streamColor.destroy();
		// 关闭设备
		m_xtion.close();
		// 最后关闭OpenNI
		openni::OpenNI::shutdown();
	}
	

}

RGBDCamera::RGBDCamera(const RGBDCamera& other)
	: m_has_device(other.m_has_device),
	m_rgb_cam_id(other.m_rgb_cam_id),
	m_rc(other.m_rc),
	m_xtion(),  // 复制 openni::Device 对象
	m_streamDepth(),  // 复制 openni::VideoStream 对象
	camera_model(other.camera_model),
	utility(other.utility),
	m_cvt(other.m_cvt),
	m_rgb(other.m_rgb.clone()),  // 复制 cv::Mat 对象
	m_depth(other.m_depth.clone())  // 复制 cv::Mat 对象
{
}

void RGBDCamera::showdevice()
{
	if (!m_has_device) { std::cout << "No device was acquired!\n"; return; } // 先判断有没有设备

	// 获取设备信息  
	openni::Array<openni::DeviceInfo> aDeviceList;
	openni::OpenNI::enumerateDevices(&aDeviceList);

	std::cout << "电脑上连接着 " << aDeviceList.getSize() << " 个体感设备." << "\n";

	for (int i = 0; i < aDeviceList.getSize(); ++i)
	{
		std::cout << "设备 " << i << "\n";
		const openni::DeviceInfo& rDevInfo = aDeviceList[i];
		std::cout << "设备名： " << rDevInfo.getName() << "\n";
		std::cout << "设备Id： " << rDevInfo.getUsbProductId() << "\n";
		std::cout << "供应商名： " << rDevInfo.getVendor() << "\n";
		std::cout << "供应商Id: " << rDevInfo.getUsbVendorId() << "\n";
		std::cout << "设备URI: " << rDevInfo.getUri() << "\n";

	}
}

void RGBDCamera::get_img_from_cam()
{
	if (!m_has_device) { std::cout << "No device was acquired!\n"; return;} // 先判断有没有设备


	// 读取RGB帧
	cv::Mat frame;
	m_imgCap >> frame;

	// 检查RGB帧是否成功读取
	if (frame.empty()) {
		std::cerr << "Error: Failed to capture RGB frame.\n";
		return;
	}

	//m_rgb = frame.clone();



	// 读取数据流
	m_rc = m_streamDepth.readFrame(&m_frameDepth);
	//cv::Mat mImageDepth, hImageDepth;
	cv::Mat hImageDepth;

	if (m_rc == openni::STATUS_OK)
	{
		// 将深度数据转换成OpenCV格式
		cv::Mat depthtemp(m_frameDepth.getHeight(), m_frameDepth.getWidth(), CV_16UC1, (void*)m_frameDepth.getData()); //CV_16UC1
		//mImageDepth = depthtemp;

		//hMirrorTrans(mImageDepth, hImageDepth);
		// 镜像翻转一下得到的深度图，这样便于后续对齐
		camera_model.hMirrorTrans(depthtemp, hImageDepth);
		m_depth = hImageDepth.clone();
	}


	//// 读取RGB帧
	//cv::Mat frame;
	//m_imgCap >> frame;

	//// 检查RGB帧是否成功读取
	//if (frame.empty()) {
	//	std::cerr << "Error: Failed to capture RGB frame.\n";
	//	return;
	//}

	//m_rgb = frame.clone();
	m_rgb = frame.clone();


}


PyObject* RGBDCamera::read_rgb_img(const std::string& path)
{
	cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
	//Mat -> Numpy
	NDArrayConverter cvt;
	PyObject* ret = cvt.toNDArray(img);
	return ret;
}

PyObject* RGBDCamera::read_depth_img(const std::string& path)
{
	cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
	//Mat -> Numpy
	//NDArrayConverter cvt;
	PyObject* ret = m_cvt.toNDArray(img);
	return ret;
}

void RGBDCamera::registration_existingimg(const std::string& rgb_file_name, const std::string& depth_file_name)
{
	// 读取RGB和深度图
	cv::Mat rgb = cv::imread(rgb_file_name);
	cv::Mat depth = cv::imread(depth_file_name, cv::IMREAD_UNCHANGED);

	// 通过深度图与深度相机下像素坐标与空间点映射关系（可以看作一条条射线），来获得深度图像素对应深度目坐标系下空间点
	cv::Mat points = camera_model.Get3DPoints(depth, camera_model.depth_pixels_to_points);

	// 上述坐标点转换至RGB目坐标系下
	cv::Mat T_extended = cv::repeat(camera_model.T_depth2rgb, 1, camera_model.height * camera_model.width); // [3, 1] -> [3, H*W]
	cv::Mat points_inrgb = camera_model.R_depth2rgb * points + T_extended; // 转换关键步骤

	// 获得空间点在rgb下的坐标位置
	cv::Mat depth_inrgb = camera_model.GetPixels(points_inrgb, camera_model.RGBCameraMatrix, depth);

	// 转换至rgb坐标系下，方便构建点云
	cv::Mat points_rgbcoord = camera_model.PixelsCoordTransfer(points_inrgb);

	// 中值滤波处理一下
	cv::medianBlur(depth_inrgb, depth_inrgb, 5);
	cv::medianBlur(points_rgbcoord, points_rgbcoord, 5);

	// 寻找最小值和最大值
	double minVal, maxVal;
	cv::minMaxLoc(depth_inrgb, &minVal, &maxVal);

	// 将深度图归一化到0-255范围，便于显示
	float scale = 255.0 / static_cast<float>((maxVal - minVal));
	float shift = static_cast<float>(-minVal) * scale;

	cv::Mat depth_inrgb_CV8U;
	depth_inrgb.convertTo(depth_inrgb_CV8U, CV_8U, scale, shift);


	// 将深度图转换为三通道，以便与 RGB 图像叠加
	cv::Mat depth_inrgb_color;
	cv::applyColorMap(depth_inrgb_CV8U, depth_inrgb_color, cv::COLORMAP_JET);

	// 叠加深度图+rgb图像
	cv::Mat rgb_depth;
	double depthweight = 0.5;
	//cv::addWeighted(depth_inrgb_color, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
	cv::addWeighted(depth_inrgb_color, depthweight, rgb, (1 - depthweight), 0.0, rgb_depth);

	// 保存到结构体，便于python接口提取
	m_rgb = rgb.clone();
	m_depth = depth.clone();
	m_points_inrgb = points_inrgb.clone();
	m_depth_inrgb = depth_inrgb.clone();
	m_points_rgbcoord = points_rgbcoord.clone();
	m_depth_inrgb_CV8U = depth_inrgb_CV8U.clone();
	m_depth_inrgb_color = depth_inrgb_color.clone();
	m_rgb_depth = rgb_depth.clone();

	return;

}

void RGBDCamera::registration_capturedimg()
{
	// 通过深度图与深度相机下像素坐标与空间点映射关系（可以看作一条条射线），来获得深度图像素对应深度目坐标系下空间点
	cv::Mat points = camera_model.Get3DPoints(m_depth, camera_model.depth_pixels_to_points);

	// 上述坐标点转换至RGB目坐标系下
	cv::Mat T_extended = cv::repeat(camera_model.T_depth2rgb, 1, camera_model.height * camera_model.width); // [3, 1] -> [3, H*W]
	cv::Mat points_inrgb = camera_model.R_depth2rgb * points + T_extended; // 转换关键步骤

	// 获得空间点在rgb下的坐标位置
	cv::Mat depth_inrgb = camera_model.GetPixels(points_inrgb, camera_model.RGBCameraMatrix, m_depth);

	// 转换至rgb坐标系下，方便构建点云
	cv::Mat points_rgbcoord = camera_model.PixelsCoordTransfer(points_inrgb);

	// 中值滤波处理一下
	cv::medianBlur(depth_inrgb, depth_inrgb, 5);
	cv::medianBlur(points_rgbcoord, points_rgbcoord, 5);

	// 寻找最小值和最大值
	double minVal, maxVal;
	cv::minMaxLoc(depth_inrgb, &minVal, &maxVal);

	// 将深度图归一化到0-255范围，便于显示
	float scale = 255.0 / static_cast<float>((maxVal - minVal));
	float shift = static_cast<float>(-minVal) * scale;

	cv::Mat depth_inrgb_CV8U;
	depth_inrgb.convertTo(depth_inrgb_CV8U, CV_8U, scale, shift);


	// 将深度图转换为三通道，以便与 RGB 图像叠加
	cv::Mat depth_inrgb_color;
	cv::applyColorMap(depth_inrgb_CV8U, depth_inrgb_color, cv::COLORMAP_JET);

	// 叠加深度图+rgb图像
	cv::Mat rgb_depth;
	double depthweight = 0.5;
	//cv::addWeighted(depth_inrgb_color, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
	cv::addWeighted(depth_inrgb_color, depthweight, m_rgb, (1 - depthweight), 0.0, rgb_depth);

	// 保存到结构体，便于python接口提取
	//m_rgb = rgb.clone();
	//m_depth = depth.clone();
	m_points_inrgb = points_inrgb.clone();
	m_depth_inrgb = depth_inrgb.clone();
	m_points_rgbcoord = points_rgbcoord.clone();
	m_depth_inrgb_CV8U = depth_inrgb_CV8U.clone();
	m_depth_inrgb_color = depth_inrgb_color.clone();
	m_rgb_depth = rgb_depth.clone();
}

PyObject* RGBDCamera::get_mat(const std::string& mat_name)
{
	cv::Mat temp;
	if		(mat_name == "rgb")					{ temp = m_rgb; }
	else if (mat_name == "depth")				{ temp = m_depth; }
	else if (mat_name == "points_inrgb")		{ temp = m_points_inrgb; }
	else if (mat_name == "depth_inrgb")			{ temp = m_depth_inrgb; }
	else if (mat_name == "points_rgbcoord")		{ temp = m_points_rgbcoord; }
	else if (mat_name == "depth_inrgb_CV8U")	{ temp = m_depth_inrgb_CV8U; }
	else if (mat_name == "depth_inrgb_color")	{ temp = m_depth_inrgb_color; }
	else if (mat_name == "rgb_depth")			{ temp = m_rgb_depth; }
	else if (mat_name == "rgb_drawn")			{ temp = m_rgb_drawn; }
	else { std::cout << "no member is called " << mat_name; return nullptr; }

	PyObject* ret = m_cvt.toNDArray(temp);
	return ret;
}

//PyObject* RGBDCamera::get_feature_points_3D(const std::vector<uint16_t>& feature_x, const std::vector<uint16_t>& feature_y)
PyObject* RGBDCamera::get_feature_points_3D_6(const boost::python::list& feature_x, const boost::python::list& feature_y, bool draw)
{
	int f_num = boost::python::len(feature_x);
	//int f_num = feature_x.size();

	// m_feature_points_now尺寸为3*n，其中3代表xyz坐标，n代表有n个特征点，初始化默认为3*6
	// 如果n!=6，则会resize
	if (f_num != m_feature_points_now.cols)
	{
		cv::Size targetSize(3, f_num);
		cv::resize(m_feature_points_now, m_feature_points_now, targetSize);
	}

	m_rgb_drawn = m_rgb.clone();

	for (int i = 0; i < f_num; i++)
	{
		//m_feature_points_now.at<float>(0, i) = m_points_rgbcoord.at<cv::Vec3f>(feature_y[i], feature_x[i])[0];
		//m_feature_points_now.at<float>(1, i) = m_points_rgbcoord.at<cv::Vec3f>(feature_y[i], feature_x[i])[1];
		//m_feature_points_now.at<float>(2, i) = m_points_rgbcoord.at<cv::Vec3f>(feature_y[i], feature_x[i])[2];
		
		int x_i = boost::python::extract<int>(feature_x[i]);
		int y_i = boost::python::extract<int>(feature_y[i]);

		m_feature_points_now.at<float>(0, i) = m_points_rgbcoord.at<cv::Vec3f>(y_i, x_i)[0];
		m_feature_points_now.at<float>(1, i) = m_points_rgbcoord.at<cv::Vec3f>(y_i, x_i)[1];
		m_feature_points_now.at<float>(2, i) = m_points_rgbcoord.at<cv::Vec3f>(y_i, x_i)[2];


		// 判断是不是转换前深度为0的点，如果是，则xyz均记为0，方便后续处理
		if (std::abs(m_feature_points_now.at<float>(2, i) - m_nodepth_point_inrgb.at<float>(2, 0)) < 1e-4)
		{
			//std::cout << x << " " << y << '\n';
			//std::cout << depth_inrgb.at<float>(y, x) << '\n';
			m_feature_points_now.at<float>(0, i) = 0.0f;
			m_feature_points_now.at<float>(1, i) = 0.0f;
			m_feature_points_now.at<float>(2, i) = -1.0f;
		}

		// 对m_rgb_drawn绘制特征点位置
		if (draw)
		{
			cv::Point feature(x_i, y_i);
			cv::circle(m_rgb_drawn, feature, 3, cv::Scalar(0, 0, 255), -1); // 红色点，半径为3
		}
	}

	
	PyObject* ret = m_cvt.toNDArray(m_feature_points_now.clone());
	return ret;

}

PyObject* RGBDCamera::get_feature_points_3D(const boost::python::list& feature_x, const boost::python::list& feature_y, bool draw)
{
	int f_num = boost::python::len(feature_x);
	cv::Mat feature_points_now(3, f_num, CV_32F);

	m_rgb_drawn = m_rgb.clone();

	for (int i = 0; i < f_num; i++)
	{
		//m_feature_points_now.at<float>(0, i) = m_points_rgbcoord.at<cv::Vec3f>(feature_y[i], feature_x[i])[0];
		//m_feature_points_now.at<float>(1, i) = m_points_rgbcoord.at<cv::Vec3f>(feature_y[i], feature_x[i])[1];
		//m_feature_points_now.at<float>(2, i) = m_points_rgbcoord.at<cv::Vec3f>(feature_y[i], feature_x[i])[2];

		int x_i = boost::python::extract<int>(feature_x[i]);
		int y_i = boost::python::extract<int>(feature_y[i]);

		feature_points_now.at<float>(0, i) = m_points_rgbcoord.at<cv::Vec3f>(y_i, x_i)[0];
		feature_points_now.at<float>(1, i) = m_points_rgbcoord.at<cv::Vec3f>(y_i, x_i)[1];
		feature_points_now.at<float>(2, i) = m_points_rgbcoord.at<cv::Vec3f>(y_i, x_i)[2];


		// 判断是不是转换前深度为0的点，如果是，则xyz均记为0，方便后续处理
		if (std::abs(feature_points_now.at<float>(2, i) - m_nodepth_point_inrgb.at<float>(2, 0)) < 1e-4)
		{
			//std::cout << x << " " << y << '\n';
			//std::cout << depth_inrgb.at<float>(y, x) << '\n';
			feature_points_now.at<float>(0, i) = 0.0f;
			feature_points_now.at<float>(1, i) = 0.0f;
			feature_points_now.at<float>(2, i) = -1.0f;
		}

		// 对m_rgb_drawn绘制特征点位置
		
		if (draw)
		{
			cv::Point feature(x_i, y_i);
			cv::circle(m_rgb_drawn, feature, 3, cv::Scalar(0, 0, 255), -1); // 红色点，半径为3
		}	
	}


	PyObject* ret = m_cvt.toNDArray(feature_points_now.clone());
	return ret;
}

PyObject* RGBDCamera::get_pose_6p(bool draw)
{
	// 如果存在异常数据，直接全部赋值为-1，返回
	for (int i = 0; i < 6; i++)
	{
		if (std::abs(m_feature_points_now.at<float>(2, i) - (-1.0f)) < 1e-4)
		{
			m_motion_now.at<float>(0, 0) = -1.0f;
			m_motion_now.at<float>(1, 0) = -1.0f;
			m_motion_now.at<float>(2, 0) = -1.0f;
			m_motion_now.at<float>(3, 0) = -1.0f;
			m_motion_now.at<float>(4, 0) = -1.0f;
			m_motion_now.at<float>(5, 0) = -1.0f;

			PyObject* ret = m_cvt.toNDArray(m_motion_now.clone());
			return ret;
		}

	}

	// 正常处理
	cv::Mat p1 = (m_feature_points_now.col(1) + m_feature_points_now.col(2)) / 2;
	cv::Mat p2 = (m_feature_points_now.col(3) + m_feature_points_now.col(4)) / 2;
	cv::Mat p3 = m_feature_points_now.col(5);
	cv::Mat p4 = m_feature_points_now.col(0);
			
	cv::Mat motion = utility.PositionToMotion(p1, p2, p3, p4);
	m_motion_now = motion;
	if (draw) {
		utility.DrawCoord(m_rgb_drawn, camera_model.RGBCameraMatrix, p1, p2, p3, p4);
	}	

	PyObject* ret = m_cvt.toNDArray(m_motion_now.clone());
	return ret;

}





BOOST_PYTHON_MODULE(BoostPythonCam) {
	using namespace boost::python;
	Py_Initialize();
	np::initialize();

	boost::python::class_<RGBDCamera>("RGBDCamera", boost::python::init<>())
	//boost::python::class_<RGBDCamera>("RGBDCamera", boost::python::init<const std::string&, const bool&, const int&>());
		.def(boost::python::init<const std::string&, bool, int>())
		.def("showdevice", &RGBDCamera::showdevice)
		.def("get_img_from_cam", &RGBDCamera::get_img_from_cam)
		.def("read_rgb_img", &RGBDCamera::read_rgb_img)
		.def("read_depth_img", &RGBDCamera::read_depth_img)
		.def("registration_capturedimg", &RGBDCamera::registration_capturedimg)
		.def("registration_existingimg", &RGBDCamera::registration_existingimg)
		.def("get_mat", &RGBDCamera::get_mat)
		.def("get_feature_points_3D_6", &RGBDCamera::get_feature_points_3D_6)
		.def("get_feature_points_3D", &RGBDCamera::get_feature_points_3D)
		.def("get_pose_6p", &RGBDCamera::get_pose_6p);


}