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
	// �������ص���Ӧ�ռ������ӳ���ϵ
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

	// ��¼ԭʼdepth��Ϊ0�ĵ���rgb����ϵ�¶�Ӧ�����꣬�����޳��쳣����
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
	// �������ص���Ӧ�ռ������ӳ���ϵ
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

	// ��¼ԭʼdepth��Ϊ0�ĵ���rgb����ϵ�¶�Ӧ�����꣬�����޳��쳣����
	cv::Mat nodepth_point = (cv::Mat_<float>(3, 1) << 0.0f, 0.0f, 0.0f);
	m_nodepth_point_inrgb = camera_model.R_depth2rgb * nodepth_point + camera_model.T_depth2rgb;

	m_feature_points_now.create(3, 6, CV_32F);
	m_motion_now.create(6, 6, CV_32F);

	m_rgb.create(camera_model.height, camera_model.width, CV_8UC3);
	m_depth.create(camera_model.height, camera_model.width, CV_16UC1);

	// has_device == true: ������ɼ���غ�����ʼ��
	m_rgb_cam_id = rgb_cam_id;
	m_has_device = has_device;
	if (m_has_device == true)
	{
		m_rc = openni::STATUS_OK;

		// ��ʼ��OpenNI����
		openni::OpenNI::initialize();

		showdevice();

		// ��������Device�豸��
		//openni::Device xtion;
		const char* deviceURL = openni::ANY_DEVICE;  //�豸��
		m_rc = m_xtion.open(deviceURL);


		// �������������
		m_rc = m_streamDepth.create(m_xtion, openni::SENSOR_DEPTH);
		if (m_rc == openni::STATUS_OK)
		{
			// �������ͼ����Ƶģʽ
			openni::VideoMode mModeDepth;
			// �ֱ��ʴ�С
			mModeDepth.setResolution(640, 480);
			// ÿ��30֡
			mModeDepth.setFps(30);
			// ���ظ�ʽ
			mModeDepth.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);

			m_streamDepth.setVideoMode(mModeDepth);

			int nFilterEnable = 0;
			int dataSize = 4;
			//streamDepth.setProperty(XN_STREAM_PROPERTY_HOLE_FILTER, (uint8_t*)&nFilterEnable, dataSize);

			// �����������
			m_rc = m_streamDepth.start();
			if (m_rc != openni::STATUS_OK)
			{
				std::cout << "�޷��������������" << openni::OpenNI::getExtendedError() << "\n";
				m_streamDepth.destroy();
			}
		}
		else
		{
			std::cout << "�޷����������������" << openni::OpenNI::getExtendedError() << "\n";
		}




		// ����RGB����ɼ���
		m_imgCap.open(m_rgb_cam_id);

		// �������ͷ�Ƿ�ɹ���
		if (!m_imgCap.isOpened()) {
			std::cerr << "Error: Unable to open camera." << std::endl;
			return;
		}

		// ��������ͷ��������ѡ��
		// ���磬���÷ֱ���
		m_imgCap.set(cv::CAP_PROP_FRAME_WIDTH, camera_model.width);
		m_imgCap.set(cv::CAP_PROP_FRAME_HEIGHT, camera_model.height);

	}
	
}

RGBDCamera::~RGBDCamera()
{
	if (m_has_device)
	{
		// �ر�������
		m_streamDepth.destroy();
		//streamColor.destroy();
		// �ر��豸
		m_xtion.close();
		// ���ر�OpenNI
		openni::OpenNI::shutdown();
	}
	

}

RGBDCamera::RGBDCamera(const RGBDCamera& other)
	: m_has_device(other.m_has_device),
	m_rgb_cam_id(other.m_rgb_cam_id),
	m_rc(other.m_rc),
	m_xtion(),  // ���� openni::Device ����
	m_streamDepth(),  // ���� openni::VideoStream ����
	camera_model(other.camera_model),
	utility(other.utility),
	m_cvt(other.m_cvt),
	m_rgb(other.m_rgb.clone()),  // ���� cv::Mat ����
	m_depth(other.m_depth.clone())  // ���� cv::Mat ����
{
}

void RGBDCamera::showdevice()
{
	if (!m_has_device) { std::cout << "No device was acquired!\n"; return; } // ���ж���û���豸

	// ��ȡ�豸��Ϣ  
	openni::Array<openni::DeviceInfo> aDeviceList;
	openni::OpenNI::enumerateDevices(&aDeviceList);

	std::cout << "������������ " << aDeviceList.getSize() << " ������豸." << "\n";

	for (int i = 0; i < aDeviceList.getSize(); ++i)
	{
		std::cout << "�豸 " << i << "\n";
		const openni::DeviceInfo& rDevInfo = aDeviceList[i];
		std::cout << "�豸���� " << rDevInfo.getName() << "\n";
		std::cout << "�豸Id�� " << rDevInfo.getUsbProductId() << "\n";
		std::cout << "��Ӧ������ " << rDevInfo.getVendor() << "\n";
		std::cout << "��Ӧ��Id: " << rDevInfo.getUsbVendorId() << "\n";
		std::cout << "�豸URI: " << rDevInfo.getUri() << "\n";

	}
}

void RGBDCamera::get_img_from_cam()
{
	if (!m_has_device) { std::cout << "No device was acquired!\n"; return;} // ���ж���û���豸


	// ��ȡRGB֡
	cv::Mat frame;
	m_imgCap >> frame;

	// ���RGB֡�Ƿ�ɹ���ȡ
	if (frame.empty()) {
		std::cerr << "Error: Failed to capture RGB frame.\n";
		return;
	}

	//m_rgb = frame.clone();



	// ��ȡ������
	m_rc = m_streamDepth.readFrame(&m_frameDepth);
	//cv::Mat mImageDepth, hImageDepth;
	cv::Mat hImageDepth;

	if (m_rc == openni::STATUS_OK)
	{
		// ���������ת����OpenCV��ʽ
		cv::Mat depthtemp(m_frameDepth.getHeight(), m_frameDepth.getWidth(), CV_16UC1, (void*)m_frameDepth.getData()); //CV_16UC1
		//mImageDepth = depthtemp;

		//hMirrorTrans(mImageDepth, hImageDepth);
		// ����תһ�µõ������ͼ���������ں�������
		camera_model.hMirrorTrans(depthtemp, hImageDepth);
		m_depth = hImageDepth.clone();
	}


	//// ��ȡRGB֡
	//cv::Mat frame;
	//m_imgCap >> frame;

	//// ���RGB֡�Ƿ�ɹ���ȡ
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
	// ��ȡRGB�����ͼ
	cv::Mat rgb = cv::imread(rgb_file_name);
	cv::Mat depth = cv::imread(depth_file_name, cv::IMREAD_UNCHANGED);

	// ͨ�����ͼ��������������������ռ��ӳ���ϵ�����Կ���һ�������ߣ�����������ͼ���ض�Ӧ���Ŀ����ϵ�¿ռ��
	cv::Mat points = camera_model.Get3DPoints(depth, camera_model.depth_pixels_to_points);

	// ���������ת����RGBĿ����ϵ��
	cv::Mat T_extended = cv::repeat(camera_model.T_depth2rgb, 1, camera_model.height * camera_model.width); // [3, 1] -> [3, H*W]
	cv::Mat points_inrgb = camera_model.R_depth2rgb * points + T_extended; // ת���ؼ�����

	// ��ÿռ����rgb�µ�����λ��
	cv::Mat depth_inrgb = camera_model.GetPixels(points_inrgb, camera_model.RGBCameraMatrix, depth);

	// ת����rgb����ϵ�£����㹹������
	cv::Mat points_rgbcoord = camera_model.PixelsCoordTransfer(points_inrgb);

	// ��ֵ�˲�����һ��
	cv::medianBlur(depth_inrgb, depth_inrgb, 5);
	cv::medianBlur(points_rgbcoord, points_rgbcoord, 5);

	// Ѱ����Сֵ�����ֵ
	double minVal, maxVal;
	cv::minMaxLoc(depth_inrgb, &minVal, &maxVal);

	// �����ͼ��һ����0-255��Χ��������ʾ
	float scale = 255.0 / static_cast<float>((maxVal - minVal));
	float shift = static_cast<float>(-minVal) * scale;

	cv::Mat depth_inrgb_CV8U;
	depth_inrgb.convertTo(depth_inrgb_CV8U, CV_8U, scale, shift);


	// �����ͼת��Ϊ��ͨ�����Ա��� RGB ͼ�����
	cv::Mat depth_inrgb_color;
	cv::applyColorMap(depth_inrgb_CV8U, depth_inrgb_color, cv::COLORMAP_JET);

	// �������ͼ+rgbͼ��
	cv::Mat rgb_depth;
	double depthweight = 0.5;
	//cv::addWeighted(depth_inrgb_color, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
	cv::addWeighted(depth_inrgb_color, depthweight, rgb, (1 - depthweight), 0.0, rgb_depth);

	// ���浽�ṹ�壬����python�ӿ���ȡ
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
	// ͨ�����ͼ��������������������ռ��ӳ���ϵ�����Կ���һ�������ߣ�����������ͼ���ض�Ӧ���Ŀ����ϵ�¿ռ��
	cv::Mat points = camera_model.Get3DPoints(m_depth, camera_model.depth_pixels_to_points);

	// ���������ת����RGBĿ����ϵ��
	cv::Mat T_extended = cv::repeat(camera_model.T_depth2rgb, 1, camera_model.height * camera_model.width); // [3, 1] -> [3, H*W]
	cv::Mat points_inrgb = camera_model.R_depth2rgb * points + T_extended; // ת���ؼ�����

	// ��ÿռ����rgb�µ�����λ��
	cv::Mat depth_inrgb = camera_model.GetPixels(points_inrgb, camera_model.RGBCameraMatrix, m_depth);

	// ת����rgb����ϵ�£����㹹������
	cv::Mat points_rgbcoord = camera_model.PixelsCoordTransfer(points_inrgb);

	// ��ֵ�˲�����һ��
	cv::medianBlur(depth_inrgb, depth_inrgb, 5);
	cv::medianBlur(points_rgbcoord, points_rgbcoord, 5);

	// Ѱ����Сֵ�����ֵ
	double minVal, maxVal;
	cv::minMaxLoc(depth_inrgb, &minVal, &maxVal);

	// �����ͼ��һ����0-255��Χ��������ʾ
	float scale = 255.0 / static_cast<float>((maxVal - minVal));
	float shift = static_cast<float>(-minVal) * scale;

	cv::Mat depth_inrgb_CV8U;
	depth_inrgb.convertTo(depth_inrgb_CV8U, CV_8U, scale, shift);


	// �����ͼת��Ϊ��ͨ�����Ա��� RGB ͼ�����
	cv::Mat depth_inrgb_color;
	cv::applyColorMap(depth_inrgb_CV8U, depth_inrgb_color, cv::COLORMAP_JET);

	// �������ͼ+rgbͼ��
	cv::Mat rgb_depth;
	double depthweight = 0.5;
	//cv::addWeighted(depth_inrgb_color, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
	cv::addWeighted(depth_inrgb_color, depthweight, m_rgb, (1 - depthweight), 0.0, rgb_depth);

	// ���浽�ṹ�壬����python�ӿ���ȡ
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

	// m_feature_points_now�ߴ�Ϊ3*n������3����xyz���꣬n������n�������㣬��ʼ��Ĭ��Ϊ3*6
	// ���n!=6�����resize
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


		// �ж��ǲ���ת��ǰ���Ϊ0�ĵ㣬����ǣ���xyz����Ϊ0�������������
		if (std::abs(m_feature_points_now.at<float>(2, i) - m_nodepth_point_inrgb.at<float>(2, 0)) < 1e-4)
		{
			//std::cout << x << " " << y << '\n';
			//std::cout << depth_inrgb.at<float>(y, x) << '\n';
			m_feature_points_now.at<float>(0, i) = 0.0f;
			m_feature_points_now.at<float>(1, i) = 0.0f;
			m_feature_points_now.at<float>(2, i) = -1.0f;
		}

		// ��m_rgb_drawn����������λ��
		if (draw)
		{
			cv::Point feature(x_i, y_i);
			cv::circle(m_rgb_drawn, feature, 3, cv::Scalar(0, 0, 255), -1); // ��ɫ�㣬�뾶Ϊ3
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


		// �ж��ǲ���ת��ǰ���Ϊ0�ĵ㣬����ǣ���xyz����Ϊ0�������������
		if (std::abs(feature_points_now.at<float>(2, i) - m_nodepth_point_inrgb.at<float>(2, 0)) < 1e-4)
		{
			//std::cout << x << " " << y << '\n';
			//std::cout << depth_inrgb.at<float>(y, x) << '\n';
			feature_points_now.at<float>(0, i) = 0.0f;
			feature_points_now.at<float>(1, i) = 0.0f;
			feature_points_now.at<float>(2, i) = -1.0f;
		}

		// ��m_rgb_drawn����������λ��
		
		if (draw)
		{
			cv::Point feature(x_i, y_i);
			cv::circle(m_rgb_drawn, feature, 3, cv::Scalar(0, 0, 255), -1); // ��ɫ�㣬�뾶Ϊ3
		}	
	}


	PyObject* ret = m_cvt.toNDArray(feature_points_now.clone());
	return ret;
}

PyObject* RGBDCamera::get_pose_6p(bool draw)
{
	// ��������쳣���ݣ�ֱ��ȫ����ֵΪ-1������
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

	// ��������
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