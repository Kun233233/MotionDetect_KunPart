#ifndef BOOST_PYTHON_CAM_H
#define BOOST_PYTHON_CAM_H

#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
# include <boost/python.hpp>
# include <boost/python/module.hpp>
# include <boost/python/def.hpp>
# include <boost/python/numpy.hpp>
# include <opencv2/opencv.hpp>
# include <conversion.h>
# include <nlohmann/json.hpp>

# include <CameraModel.h>
# include <Utility.h>

#include <iostream>
#include <fstream>


class RGBDCamera
{
public:
	RGBDCamera();
	RGBDCamera(const std::string& param_path);
	~RGBDCamera();


	PyObject* read_rgb_img(const std::string& path);
	PyObject* read_depth_img(const std::string& path);
	void registration_existingimg(const std::string& rgb_file_name, const std::string& depth_file_name);
	PyObject* get_mat(const std::string& mat_name);

protected:
	int height;
	int width;

	CameraModel camera_model; // RGBD相机相关参数和函数实现
	Utility utility; // 其他一些工具处理函数
	NDArrayConverter m_cvt; // 便于实现c++ cv::Mat 和 python numpy.array 的转换

	cv::Mat m_rgb;
	cv::Mat m_depth;
	cv::Mat m_points_inrgb;
	cv::Mat m_depth_inrgb;
	cv::Mat m_points_rgbcoord;
	cv::Mat m_depth_inrgb_CV8U;
	cv::Mat m_depth_inrgb_color;
	cv::Mat m_rgb_depth;


};






#endif // !BOOST_PYTHON_CAM_H

