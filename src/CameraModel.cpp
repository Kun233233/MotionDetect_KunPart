#include "CameraModel.h"

CameraModel::CameraModel()
{
	//this->max_depth = max_depth;
	//this->square_size = 12.0f;
	//this->square_size = 10.0f; // 新标定板
	this->square_size = 1.0f; // 新标定板


	// 对于两个相机的参数，目前先写死，后续可以改为读取外部文件获取
	// RGB相机内外参
	//this->RGBCameraMatrix = (cv::Mat_<float>(3, 3) << 597.4286735057399, 0, 356.4896812646391,
	//													0, 590.3135817242555, 265.6565473501195,
	//													0, 0, 1);
	//this->RGBCameraMatrix_inv = RGBCameraMatrix.inv();
	//this->RGBDistCoeffs = (cv::Mat_<float>(1, 5) << 0.02433779150902952, 0.1806910557398652, 0.01504073394057258, 0.01982505451991676, -0.4655100996385541);
	//this->RGBRotVec = (cv::Mat_<float>(1, 3) << 0.1264627521012061, 0.5634184176200842, 1.660489725237417);
	//this->RGBTransVec = (cv::Mat_<float>(1, 3) << 3.294513374873486, -4.191418478429084, 20.54231028621967);

	// 新标定板版本 RGB
	this->RGBCameraMatrix = (cv::Mat_<float>(3, 3) << 607.8374395250249, 0, 324.8353393347486,
		0, 607.2444348128537, 237.149453913389,
		0, 0, 1);
	this->RGBCameraMatrix_inv = RGBCameraMatrix.inv();
	this->RGBDistCoeffs = (cv::Mat_<float>(1, 5) << -0.03196269902935066, 0.9470628443387368, -0.001401899373235646, -0.0003753770047383805, -2.547322490761638);
	this->RGBRotVec = (cv::Mat_<float>(1, 3) << -0.02968701824799445, 0.002282037274796454, -1.518336358197425);
	this->RGBTransVec = (cv::Mat_<float>(1, 3) << -34.90894696703963, 56.47315618102284, 323.6092183950838);
	//this->RGBTransVec = (cv::Mat_<float>(1, 3) << -50.90894696703963, 56.47315618102284, 323.6092183950838);

	cv::Rodrigues(this->RGBRotVec, this->RGBRotationMat);

	//// 深度相机内外参
	//this->DepthCameraMatrix = (cv::Mat_<float>(3, 3) << 569.1382523087108, 0, 330.4704844461951,
	//	0, 564.5139460154893, 250.400178575307,
	//	0, 0, 1);
	//this->DepthCameraMatrix_inv = DepthCameraMatrix.inv();
	//this->DepthDistCoeffs = (cv::Mat_<float>(1, 5) << -0.1802622269847234, 0.9963006566582099, -0.001074564774769674, 0.002966307587880594, -2.140745337976587);
	//this->DepthRotVec = (cv::Mat_<float>(1, 3) << 0.1313875859188382, 0.62437610031627, 1.648945446919959);
	//this->DepthTransVec = (cv::Mat_<float>(1, 3) << 6.166359975994443, -3.53947047998281, 20.74186807903174);
	
	// 新标定板版本 Depth
	this->DepthCameraMatrix = (cv::Mat_<float>(3, 3) << 576.6390327180507, 0, 324.9849290814257,
		0, 575.6560633555423, 241.80528982161,
		0, 0, 1);
	this->DepthCameraMatrix_inv = DepthCameraMatrix.inv();
	this->DepthDistCoeffs = (cv::Mat_<float>(1, 5) << -0.1313667715820261, 0.7253701684598015, -0.001511899382358423, -0.001635651373407517, -1.520364579947955);
	this->DepthRotVec = (cv::Mat_<float>(1, 3) << -0.03456143021858554, -0.007579969887891717, -1.521223589678921);
	this->DepthTransVec = (cv::Mat_<float>(1, 3) << -11.21386065738118, 56.3175734828904, 323.767980354571);
	
	
	cv::Rodrigues(this->DepthRotVec, this->DepthRotationMat);

	this->R_depth2rgb = RGBRotationMat * DepthRotationMat.inv();
	this->T_depth2rgb = (RGBTransVec.t() - R_depth2rgb * DepthTransVec.t()) * square_size;

}

CameraModel::~CameraModel()
{
}

void CameraModel::printMatrixInfo(const cv::Mat& matrix, const std::string& name)
{
	std::cout << "Matrix: " << name << "\n";
	std::cout << "Type: " << matrix.type() << "\n";
	//std::cout << "Size: " << matrix.size() << "\n";
	std::cout << "shape: " << "[" << matrix.size().height << ", " << matrix.size().width << "]" << "\n";
	std::cout << "Channels: " << matrix.channels() << "\n";
	std::cout << "Depth: " << matrix.depth() << "\n";
	std::cout << "------------------------------------\n";
}

void CameraModel::hMirrorTrans(const cv::Mat& src, cv::Mat& dst)
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

cv::Mat CameraModel::Get3DPoints(const cv::Mat& depth, const cv::Mat& pixels_to_points_map)
{
	cv::Mat points(3, IMAGE_HEIGHT_480 * IMAGE_WIDTH_640, CV_32F);

	// 构建齐次坐标
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

cv::Mat CameraModel::GetPixels(const cv::Mat& points, const cv::Mat& camera_matrix, const cv::Mat& depth_map)
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

	//cv::Mat map_x = cv::Mat::zeros(depth_map.size(), CV_32FC1);
	//cv::Mat map_y = cv::Mat::zeros(depth_map.size(), CV_32FC1);
	this->map_x = cv::Mat::zeros(depth_map.size(), CV_32FC1);
	this->map_y = cv::Mat::zeros(depth_map.size(), CV_32FC1);


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

cv::Mat CameraModel::PixelsCoordTransfer(const cv::Mat& points)
{
	cv::Mat points_x_rgb;
	cv::Mat points_x = points.row(0).reshape(1, 480);
	cv::remap(points_x, points_x_rgb, map_x, map_y, cv::INTER_LANCZOS4, cv::BORDER_REPLICATE, cv::Scalar());

	cv::Mat points_y_rgb;
	cv::Mat points_y = points.row(1).reshape(1, 480);
	cv::remap(points_y, points_y_rgb, map_x, map_y, cv::INTER_LANCZOS4, cv::BORDER_REPLICATE, cv::Scalar());

	cv::Mat points_z_rgb;
	cv::Mat points_z = points.row(2).reshape(1, 480);
	cv::remap(points_z, points_z_rgb, map_x, map_y, cv::INTER_LANCZOS4, cv::BORDER_REPLICATE, cv::Scalar());

	// 创建一个空的三通道矩阵
	cv::Mat merged;
	// 合并三个单通道矩阵为一个三通道矩阵
	cv::merge(std::vector<cv::Mat>{points_x_rgb, points_y_rgb, points_z_rgb}, merged);


	return merged;
}
