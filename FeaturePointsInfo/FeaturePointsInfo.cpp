
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
#include <sstream>
#include <regex>
#include <vector>
#include <chrono>
#include <cmath>

#include <omp.h>

//#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>


//RGB w x h
const int IMAGE_WIDTH_640 = 640;
const int  IMAGE_HEIGHT_480 = 480;
//Read data outtime
const int  UVC_TIME_OUT = 3000; //ms
const float PI = 3.1415926535;

const int feature_num = 6;


enum class FeatureID
{
	LeftL	= 37,
	LeftR	= 40,
	RightL	= 43,
	RightR	= 46, 
	NoseTop	= 34, 
	MouseTop= 32
};

const std::vector<int> feature_id_series = {	static_cast<int>(FeatureID::MouseTop),	
												static_cast<int>(FeatureID::NoseTop),
												static_cast<int>(FeatureID::LeftL), 
												static_cast<int>(FeatureID::LeftR), 
												static_cast<int>(FeatureID::RightL), 
												static_cast<int>(FeatureID::RightR)
												 };






void showdevice() {
	// 获取设备信息  
	openni::Array<openni::DeviceInfo> aDeviceList;
	openni::OpenNI::enumerateDevices(&aDeviceList);

	std::cout << "电脑上连接着 " << aDeviceList.getSize() << " 个体感设备." << endl;

	for (int i = 0; i < aDeviceList.getSize(); ++i)
	{
		cout << "设备 " << i << endl;
		const openni::DeviceInfo& rDevInfo = aDeviceList[i];
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
	std::cout << "shape: " << "[" << matrix.size().height << ", " << matrix.size().width << "]" << "\n";
	std::cout << "Channels: " << matrix.channels() << "\n";
	std::cout << "Depth: " << matrix.depth() << "\n";
	std::cout << "------------------------------------\n";
}

void hMirrorTrans(const cv::Mat& src, cv::Mat& dst)
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
		const cv::Vec3b * origal3;
		cv::Vec3b* p3;
		for (int i = 0; i < rows; i++) {
			origal3 = src.ptr<cv::Vec3b>(i);
			p3 = dst.ptr<cv::Vec3b>(i);
			for (int j = 0; j < cols; j++) {
				p3[j] = origal3[cols - 1 - j];
			}
		}
		break;
	default:
		break;
	}

}

std::string getValueAt(std::ifstream& file, int targetRow, int targetColumn, char delimiter) {
	std::string line;

	// 逐行读取文件内容
	for (int currentRow = 1; std::getline(file, line); ++currentRow) {
		if (currentRow == targetRow) {
			// 使用字符串流解析每一行的数据
			std::istringstream iss(line);
			std::string value;
			std::vector<std::string> values;

			// 使用字段分隔符拆分每一行的数据
			while (std::getline(iss, value, delimiter)) {
				values.push_back(value);
			}

			// 检查目标列是否在有效范围内
			if (targetColumn >= 1 && targetColumn <= values.size()) {
				return values[targetColumn - 1]; // 返回目标列的值
			}
			else {
				std::cerr << "Error: Column " << targetColumn << " out of range." << std::endl;
				return "";
			}
		}
	}

	// 如果文件中未找到目标行
	std::cerr << "Error: Row " << targetRow << " not found." << std::endl;
	return "";
}



// 函数将二维字符串数组存储为txt文件，使用指定分隔符
void saveToTxt(const std::vector<std::vector<std::string>>& data, const std::string& filename, char delimiter) {
	// 打开文件流
	std::ofstream outfile(filename);

	// 检查文件是否成功打开
	if (!outfile.is_open()) {
		std::cerr << "Error opening file: " << filename << std::endl;
		return;
	}

	// 遍历二维数组并将数据写入文件
	for (const auto& row : data) {
		for (size_t i = 0; i < row.size(); ++i) {
			outfile << row[i];

			// 在每个元素后面添加分隔符，除了最后一个元素
			if (i < row.size() - 1) {
				outfile << delimiter;
			}
		}
		// 换行表示新的一行
		outfile << std::endl;
	}

	// 关闭文件流
	outfile.close();
}






auto GetFeaturePointsPixels(const std::string& feature_rgb_path, std::vector<std::vector<std::string>>& feature_pixels_position, char delimiter)
{
	//// 打开文本文件
	std::ifstream file(feature_rgb_path);

	// 检查文件是否成功打开
	if (!file.is_open()) {
		std::cerr << "Error opening file" << std::endl;
		//return 1;
		exit(1);
	}


	std::string line;
	// 逐行读取文件内容
	for (int currentRow = 0; std::getline(file, line); ++currentRow) {
		// 使用字符串流解析每一行的数据
		std::istringstream iss(line);
		std::string value;
		std::vector<std::string> values;
		//std::vector<std::vector<std::string>> feature_pixels_position;
		if (currentRow == 0) { continue; }

		int currentColumn = 0;
		int nextFeature = 0;
		// 使用字段分隔符拆分每一行的数据
		while (std::getline(iss, value, delimiter)) {
			++currentColumn;

			// 如果当前列是目标列，则添加到 vector 中
			if (currentColumn == feature_id_series[nextFeature] + 1) {
				std::cout << currentRow - 1 << " " << currentColumn << " " << value << "\n";
				feature_pixels_position[nextFeature][currentRow - 1] = value;
				nextFeature++;
			}
		}


	}
	return;


}




cv::Mat Get3DPoints(const cv::Mat& depth, const cv::Mat& pixels_to_points_map)
{
	//cv::Mat points(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_64FC3);
	cv::Mat points(3, IMAGE_HEIGHT_480 * IMAGE_WIDTH_640, CV_32F);

	// 对depth进行reshape操作，操作后的大小为[1, 640*480]
	cv::Mat depth_flatten = depth.reshape(1, 1);

	// 将depth_flatten中的每个元素复制到三行
	cv::Mat depth_flatten_3 = cv::repeat(depth_flatten, 3, 1);

	//转为float
	cv::Mat depth_flatten_3_float;
	depth_flatten_3.convertTo(depth_flatten_3_float, CV_32F);

	cv::multiply(depth_flatten_3_float, pixels_to_points_map, points);

	return points;
}



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



	cv::remap(depth_map, depth_in_rgb, map_x, map_y, cv::INTER_LANCZOS4, cv::BORDER_REPLICATE, cv::Scalar());
	//（1）INTER_NEAREST——最近邻插值
	//（2）INTER_LINEAR——双线性插值（默认）
	//（3）INTER_CUBIC——双三样条插值（逾4×4像素邻域内的双三次插值）
	// (4）INTER_LANCZOS4——lanczos插值（逾8×8像素邻域的Lanczos插值）

	return depth_in_rgb;

}



cv::Mat PositionToMotion(const cv::Mat& p1, const cv::Mat& p2, const cv::Mat& p3, const cv::Mat& p4)
{
	std::cout << p1 << p2 << p3 << p4 << '\n';

	cv::Mat p2_p3 = p2 - p3;
	cv::Mat p1_p3 = p1 - p3;

	// n1, n2, n3分别代表面部建立的三个坐标轴的方向
	cv::Mat n1 = p2_p3.cross(p1_p3);
	cv::normalize(n1, n1);
	cv::Mat n2 = p2_p3 - p1_p3;
	cv::normalize(n2, n2);
	cv::Mat n3 = n1.cross(n2);

	float a = std::asinf(n2.at<float>(2, 0) / n3.at<float>(2, 0));
	float b = -std::asinf(n1.at<float>(2, 0));
	float c = std::atan2f(n1.at<float>(1, 0), n1.at<float>(0, 0));

	if (a > PI / 2) { a = PI - a; }
	if (c > PI / 2) { c = PI - c; }

	cv::Mat center_position = p4 - 80.0f * n1;
	float x = center_position.at<float>(0, 0);
	float y = center_position.at<float>(1, 0);
	float z = center_position.at<float>(2, 0);

	cv::Mat center_pose = (cv::Mat_<float>(6, 1) << x, y, z, a, b, c);


	return center_pose;
}















int main()
{
	try {

		std::string  depth_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/depth_0.avi";
		std::string  rgb_video_path = "D:/aaaLab/aaagraduate/SaveVideo/DepthSavePoll/rgb_0.avi";
		// 存储成图片形式的地址
		std::string  depth_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/DepthImgs";
		std::string  rgb_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/RGBImgs";

		////std::ifstream file("D:/aaaLab/aaagraduate/SaveVideo/src/rgb.txt");
		std::regex pattern(R"(\((\d+),(\d+)\))");

		//// 打开文本文件
		//std::ifstream feature_rgb_file("D:/aaaLab/aaagraduate/SaveVideo/src/rgb.txt");
		std::string feature_rgb_path = "D:/aaaLab/aaagraduate/SaveVideo/source/rgb.txt";

		// 打开文件
		std::ifstream rgb_count_file(rgb_folder_path + "/frame_num.txt");
		// 检查文件是否成功打开
		if (!rgb_count_file.is_open()) {
			std::cerr << "Unable to open the file." << std::endl;
			return 1;
		}
		// 读取文件中的数字
		int countmax;
		rgb_count_file >> countmax;
		// 关闭文件
		rgb_count_file.close();

		// 创建一个 vector 用于存储每个元素
		std::vector<std::vector<std::string>> feature_pixels_position(feature_num);
		// 提前分配空间
		feature_pixels_position.resize(static_cast<int>(feature_num * 1.5));
		for (auto& row : feature_pixels_position) {
			row.resize(static_cast<int>(countmax * 1.5));
		}

		// 创建一个 vector 用于存储选中特征点的每一帧空间点坐标
		std::vector<std::vector<std::string>> feature_pixels_3D(feature_num);
		// 提前分配空间
		feature_pixels_3D.resize(static_cast<int>(feature_num * 1.5));
		for (auto& row : feature_pixels_3D) {
			row.resize(static_cast<int>(countmax * 1.5));
		}

		GetFeaturePointsPixels(feature_rgb_path, feature_pixels_position, '\t');



		// RGB相机内外参
		cv::Mat RGBCameraMatrix = (cv::Mat_<float>(3, 3) << 597.4286735057399, 0, 356.4896812646391,
			0, 590.3135817242555, 265.6565473501195,
			0, 0, 1);
		cv::Mat RGBCameraMatrix_inv = RGBCameraMatrix.inv();
		cv::Mat RGBDistCoeffs = (cv::Mat_<float>(1, 5) << 0.02433779150902952, 0.1806910557398652, 0.01504073394057258, 0.01982505451991676, -0.4655100996385541);
		cv::Mat RGBRotVec = (cv::Mat_<float>(1, 3) << 0.1264627521012061, 0.5634184176200842, 1.660489725237417);
		cv::Mat RGBTransVec = (cv::Mat_<float>(1, 3) << 3.294513374873486, -4.191418478429084, 20.54231028621967);
		cv::Mat RGBRotationMat;
		cv::Rodrigues(RGBRotVec, RGBRotationMat);

		// 深度相机内外参
		cv::Mat DepthCameraMatrix = (cv::Mat_<float>(3, 3) << 569.1382523087108, 0, 330.4704844461951,
			0, 564.5139460154893, 250.400178575307,
			0, 0, 1);
		cv::Mat DepthCameraMatrix_inv = DepthCameraMatrix.inv();
		cv::Mat DepthDistCoeffs = (cv::Mat_<float>(1, 5) << -0.1802622269847234, 0.9963006566582099, -0.001074564774769674, 0.002966307587880594, -2.140745337976587);
		cv::Mat DepthRotVec = (cv::Mat_<float>(1, 3) << 0.1313875859188382, 0.62437610031627, 1.648945446919959);
		cv::Mat DepthTransVec = (cv::Mat_<float>(1, 3) << 6.166359975994443, -3.53947047998281, 20.74186807903174);
		cv::Mat DepthRotationMat;
		cv::Rodrigues(DepthRotVec, DepthRotationMat);

		// 深度相机转到RGB相机的R和T  P_rgb = R * P_ir + T
		cv::Mat R = RGBRotationMat * DepthRotationMat.inv();
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

		// 记录循环次数
		int count = 0;

		

		double fps = 30.0;
		int wait_time = static_cast<int>(1 / fps * 1000.0);


		for (int i = 0; i < countmax; i++)
		{
			auto start_time = std::chrono::high_resolution_clock::now();

			std::string rgb_file_name = rgb_folder_path + "/rgb_" + std::to_string(i) + ".png"; 
			std::string depth_file_name = depth_folder_path + "/depth_" + std::to_string(i) + ".png";
			//std::cout << rgb_file_name << "\n";
			cv::Mat rgb = cv::imread(rgb_file_name);
			cv::Mat depth = cv::imread(depth_file_name, cv::IMREAD_UNCHANGED);
			//if (!depth.empty()) {
			//	std::cout << "Image size: " << depth.size() << std::endl;
			//	std::cout << "Number of channels: " << depth.channels() << std::endl;
			//}

			//else {
			//	std::cerr << "Failed to load image." << std::endl;
			//}
			//printMatrixInfo(depth, "depth");


			// 获得深度图每个像素点对应的3D空间坐标 (x, y, z)
			cv::Mat points = Get3DPoints(depth, pixels_to_points);


			cv::Mat T_extended = cv::repeat(T, 1, IMAGE_HEIGHT_480 * IMAGE_WIDTH_640); // [3, 1] -> [3, 480*640]
			cv::Mat points_inrgb = R * points + T_extended; // points应该化成(3, 1)的样子，不急，回来改

			//cv::Mat depth_inrgb = GetPixels(points_inrgb, RGBCameraMatrix, hImageDepth);
			cv::Mat depth_inrgb = GetPixels(points_inrgb, RGBCameraMatrix, depth);

			// 中值滤波处理，先测试一下 (Kun: 2024.3.7)
			cv::medianBlur(depth_inrgb, depth_inrgb, 5);

			double max_depth_value;
			// 使用 cv::minMaxLoc 函数获取最大值和位置
			cv::minMaxLoc(depth_inrgb, nullptr, &max_depth_value, nullptr, nullptr);

			float scale_factor = 255.0 / static_cast<float>(max_depth_value);
			float offset = 0.0;
			cv::Mat depth_inrgb_CV8U;
			cv::convertScaleAbs(depth_inrgb, depth_inrgb_CV8U, scale_factor, offset);
			cv::imshow("depth_inrgb_CV8U", depth_inrgb_CV8U);


			// 将深度图归一化到0-255范围，以便与 RGB 图像叠加
			cv::Mat depth_inrgb_normalized;
			cv::normalize(depth_inrgb_CV8U, depth_inrgb_normalized, 0, 255, cv::NORM_MINMAX);


			// 将深度图转换为三通道，以便与 RGB 图像叠加
			cv::Mat depth_inrgb_color;
			cv::applyColorMap(depth_inrgb_normalized, depth_inrgb_color, cv::COLORMAP_JET);

			// 叠加深度图+rgb图像
			cv::Mat rgb_depth;
			double depthweight = 0.5;
			//cv::addWeighted(depth_inrgb_color, depthweight, frame, (1 - depthweight), 0.0, rgb_depth);
			cv::addWeighted(depth_inrgb_color, depthweight, rgb, (1 - depthweight), 0.0, rgb_depth);
			cv::imshow("Mixed", rgb_depth);

			// 显示帧
			//cv::imshow("Camera Feed", frame);
			cv::imshow("Camera Feed", rgb);


			//std::ifstream file("D:/aaaLab/aaagraduate/SaveVideo/src/rgb.txt");
			//std::string value = getValueAt(file, i + 2, 37 + 1, '\t');
			//std::cout << value << "\n";
			//file.close();
			//std::string value = elements[i];
			std::vector<cv::Mat> position;
			position.resize(feature_num * 1.5);

			for (int feature_id = 0; feature_id < feature_num; feature_id++)
			{
				std::string value = feature_pixels_position[feature_id][i];
				
				std::cout << value << '\n';
				std::smatch matches;
				int x, y;
				if (std::regex_search(value, matches, pattern)) {
					// 第一个匹配项是整个字符串，后面的是括号内的两个数字
					x = std::stoi(matches[1].str());
					y = std::stoi(matches[2].str());
					std::cout << "First Number: " << x << std::endl;
					std::cout << "Second Number: " << y << std::endl;
				}
				else {
					std::cerr << "No match found" << std::endl;
				}
				//int index = y * IMAGE_WIDTH_640 + x;
				//int index = x * IMAGE_WIDTH_640 + y;
				//int index = x * IMAGE_HEIGHT_480 + y;
				int index = y * IMAGE_WIDTH_640 + x;
				// 检查索引是否在图像范围内
				if (index >= 0 && index < points_inrgb.cols) {
					// 访问 reshape 后的图像中特定位置的像素值
					float point_x = points_inrgb.at<float>(0, index);
					float point_y = points_inrgb.at<float>(1, index);
					float point_z = points_inrgb.at<float>(2, index);
					
					std::stringstream ss; // 创建一个字符串流对象
					ss << std::fixed << std::setprecision(4); // 设置小数点精度为4位
					ss << "(" << point_x << "," << point_y << "," << point_z << ")"; // 将浮点数写入字符串流中
					std::string result = ss.str(); // 从字符串流中获取组合后的字符串
					feature_pixels_3D[feature_id][i] = result;
					std::cout << result << std::endl; // 输出结果

					cv::Mat point = (cv::Mat_<float>(3, 1) << point_x, point_y, point_z);
					position[feature_id] = point;
				}
				else {
					std::cerr << "Invalid index" << std::endl;
				}

				cv::Mat motion = PositionToMotion((position[2] + position[3]) / 2, (position[4] + position[5]) / 2, position[0], position[1]);



			}


			auto end_time = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
			int new_wait_time = wait_time - static_cast<int>(duration.count());
			//std::cout << new_wait_time << " " << wait_time << " " << duration.count() << " " << static_cast<int>(duration.count());
			
			if (new_wait_time >= 1)
			{
				cv::waitKey(new_wait_time);
			}
			else
			{
				cv::waitKey(1);
			}
			


			// 终止快捷键 ESC
			if (cv::waitKey(1) == 27)
				break;

		}
		//file.close();
		std::string feature_3D_path = "D:/aaaLab/aaagraduate/SaveVideo/src/points.txt";
		saveToTxt(feature_pixels_3D, feature_3D_path, '\t');

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










