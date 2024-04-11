#ifndef BOOST_PYTHON_CAM_H
#define BOOST_PYTHON_CAM_H

#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
# include <boost/python.hpp>
# include <boost/python/module.hpp>
# include <boost/python/def.hpp>
# include <boost/python/numpy.hpp>
#include <boost/python/list.hpp>
# include <opencv2/opencv.hpp>
# include <conversion.h>
# include <nlohmann/json.hpp>

#include <OpenNI.h>
#include "OniSampleUtilities.h"
#include "UVC_Swapper.h"
#include "UVCSwapper.h"
#include "OBTypes.h"
#include "ObCommon.h"


# include <CameraModel.h>
# include <Utility.h>

#include <iostream>
#include <fstream>


class RGBDCamera
{
public:
	RGBDCamera();
	RGBDCamera(const std::string& param_path, bool has_device, int rgb_cam_id);
	~RGBDCamera();

	RGBDCamera::RGBDCamera(const RGBDCamera& other);


	void showdevice();
	//void create_capture_stream(const int& rgb_cam_id);
	void get_img_from_cam();


	PyObject* read_rgb_img(const std::string& path);
	PyObject* read_depth_img(const std::string& path);
	void registration_existingimg(const std::string& rgb_file_name, const std::string& depth_file_name);
	void registration_capturedimg();
	//void registration_existingimg(const boost::python::list& rgb_file_name, const boost::python::list& depth_file_name);
	PyObject* get_mat(const std::string& mat_name);

	//PyObject* get_feature_points_3D(const std::vector<uint16_t>& feature_x, const std::vector<uint16_t>& feature_y);
	PyObject* get_feature_points_3D(const boost::python::list& feature_x, const boost::python::list& feature_y);

	PyObject* get_pose_6p();


protected:
	bool m_has_device = false;
	int m_rgb_cam_id = -1;

	//int m_height;
	//int m_width;


	openni::Status m_rc;
	openni::Device m_xtion;
	openni::VideoStream m_streamDepth;
	cv::VideoCapture m_imgCap;
	openni::VideoFrameRef  m_frameDepth;

	CameraModel camera_model; 
	Utility utility; 
	NDArrayConverter m_cvt; 

	cv::Mat m_rgb;
	cv::Mat m_depth;
	cv::Mat m_points_inrgb;
	cv::Mat m_depth_inrgb;
	cv::Mat m_points_rgbcoord;
	cv::Mat m_depth_inrgb_CV8U;
	cv::Mat m_depth_inrgb_color;
	cv::Mat m_rgb_depth;
	cv::Mat m_nodepth_point_inrgb;

	cv::Mat m_feature_points_now;
	cv::Mat m_motion_now;
	cv::Mat m_rgb_drawn;


};






#endif // !BOOST_PYTHON_CAM_H

