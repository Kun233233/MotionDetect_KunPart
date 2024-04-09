#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB

# include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/numpy.hpp>
# include <opencv2/opencv.hpp>
#include <conversion.h>
#include "BoostPythonCam.h"


namespace np = boost::python::numpy;
using json = nlohmann::json;


//PyObject* threshold(np::ndarray const& array)
//{
//	cv::Mat img;
//	//if (array.get_dtype() != np::dtype::get_builtin<UINT8>()) {
//	//	PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
//	//	p::throw_error_already_set();
//	//}
//	//if (array.get_nd() != 2) {
//	//	PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
//	//	p::throw_error_already_set();
//	//}
//	//if (!(array.get_flags() & np::ndarray::C_CONTIGUOUS)) {
//	//	PyErr_SetString(PyExc_TypeError, "Array must be row-major contiguous");
//	//	p::throw_error_already_set();
//	//}
//	int* iter = reinterpret_cast<int*>(array.get_data());
//	int rows = array.shape(0);
//	int cols = array.shape(1);
//	img.create(cv::Size(rows, cols), CV_8U);
//	img.data = (uchar*)iter;
//
//	cv::Mat thresh;
//	cv::threshold(img, thresh, 120, 250, cv::THRESH_BINARY);
//
//	//Mat -> Numpy
//	NDArrayConverter cvt;
//	PyObject* ret = cvt.toNDArray(thresh);
//	return ret;
//}

//PyObject* read_img(const char* path)
//{
//	cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
//
//
//	//Mat -> Numpy
//	NDArrayConverter cvt;
//	PyObject* ret = cvt.toNDArray(img);
//	return ret;
//}



//const char* sayhello()
//{
//	//cv::Mat a;
//	return "hello";
//}



RGBDCamera::RGBDCamera() : camera_model(), utility(), m_cvt()
{
}

RGBDCamera::RGBDCamera(const std::string& param_path) : camera_model(param_path), utility(), m_cvt()
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
	
}

RGBDCamera::~RGBDCamera()
{

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
	else { std::cout << "no member is called " << mat_name; return nullptr; }

	PyObject* ret = m_cvt.toNDArray(temp);
	return ret;
}





BOOST_PYTHON_MODULE(BoostPythonCam) {
	using namespace boost::python;
	Py_Initialize();
	np::initialize();

	boost::python::class_<RGBDCamera>("RGBDCamera", boost::python::init<>())
		.def(boost::python::init<const std::string&>())
		.def("read_rgb_img", &RGBDCamera::read_rgb_img)
		.def("read_depth_img", &RGBDCamera::read_depth_img)
		.def("registration_existingimg", &RGBDCamera::registration_existingimg)
		.def("get_mat", &RGBDCamera::get_mat);



	//def("generateImage", generateImage);
	//def("sayhello", sayhello);
	//def("threshold", threshold);
	//def("read_img", read_img);

}