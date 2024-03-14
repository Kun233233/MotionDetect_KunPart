#include "CameraModel.h"

CameraModel::CameraModel()
{
	//this->max_depth = max_depth;
	//this->square_size = 12.0f;
	//this->square_size = 10.0f; // �±궨��
	this->square_size = 1.0f; // �±궨��


	// ������������Ĳ�����Ŀǰ��д�����������Ը�Ϊ��ȡ�ⲿ�ļ���ȡ
	// RGB��������
	//this->RGBCameraMatrix = (cv::Mat_<float>(3, 3) << 597.4286735057399, 0, 356.4896812646391,
	//													0, 590.3135817242555, 265.6565473501195,
	//													0, 0, 1);
	//this->RGBCameraMatrix_inv = RGBCameraMatrix.inv();
	//this->RGBDistCoeffs = (cv::Mat_<float>(1, 5) << 0.02433779150902952, 0.1806910557398652, 0.01504073394057258, 0.01982505451991676, -0.4655100996385541);
	//this->RGBRotVec = (cv::Mat_<float>(1, 3) << 0.1264627521012061, 0.5634184176200842, 1.660489725237417);
	//this->RGBTransVec = (cv::Mat_<float>(1, 3) << 3.294513374873486, -4.191418478429084, 20.54231028621967);

	// �±궨��汾 RGB
	this->RGBCameraMatrix = (cv::Mat_<float>(3, 3) << 604.2085370195251, 0, 322.6278592922915,
		0, 603.6937227693994, 235.528460406989,
		0, 0, 1);
	this->RGBCameraMatrix_inv = RGBCameraMatrix.inv();
	this->RGBDistCoeffs = (cv::Mat_<float>(1, 5) << -0.05016507932123436, 0.9997269173615618, -0.002050126664809821, -0.001327504815864807, -2.590438793150333);
	this->RGBRotVec = (cv::Mat_<float>(1, 3) << -0.035437177821607, 0.003173871812315727, -1.518335793783583);
	this->RGBTransVec = (cv::Mat_<float>(1, 3) << -33.72928752337317, 57.33939261054579, 321.5652351953804);

	cv::Rodrigues(this->RGBRotVec, this->RGBRotationMat);

	//// �����������
	//this->DepthCameraMatrix = (cv::Mat_<float>(3, 3) << 569.1382523087108, 0, 330.4704844461951,
	//	0, 564.5139460154893, 250.400178575307,
	//	0, 0, 1);
	//this->DepthCameraMatrix_inv = DepthCameraMatrix.inv();
	//this->DepthDistCoeffs = (cv::Mat_<float>(1, 5) << -0.1802622269847234, 0.9963006566582099, -0.001074564774769674, 0.002966307587880594, -2.140745337976587);
	//this->DepthRotVec = (cv::Mat_<float>(1, 3) << 0.1313875859188382, 0.62437610031627, 1.648945446919959);
	//this->DepthTransVec = (cv::Mat_<float>(1, 3) << 6.166359975994443, -3.53947047998281, 20.74186807903174);
	
	// �±궨��汾 Depth
	this->DepthCameraMatrix = (cv::Mat_<float>(3, 3) << 574.7477649150313, 0, 326.6927782733942,
		0, 573.8775739319774, 240.8218767510635,
		0, 0, 1);
	this->DepthCameraMatrix_inv = DepthCameraMatrix.inv();
	this->DepthDistCoeffs = (cv::Mat_<float>(1, 5) << -0.1249797724089348, 0.6554742403337946, -0.001983882261750233, -0.0009005479294426705, -1.318633314574677);
	this->DepthRotVec = (cv::Mat_<float>(1, 3) << -0.0337157754012347, -0.01083893881602162, -1.521085578010777);
	this->DepthTransVec = (cv::Mat_<float>(1, 3) << -12.18319909637389, 56.86704911511715, 322.6538039972514);
	
	
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

	// �����������
	// ��depth����reshape������������Ĵ�СΪ[1, 640*480]
	cv::Mat depth_flatten = depth.reshape(1, 1);

	// ��depth_flatten�е�ÿ��Ԫ�ظ��Ƶ�����
	cv::Mat depth_flatten_3 = cv::repeat(depth_flatten, 3, 1);

	//תΪfloat
	cv::Mat depth_flatten_3_float;
	depth_flatten_3.convertTo(depth_flatten_3_float, CV_32F);

	cv::multiply(depth_flatten_3_float, pixels_to_points_map, points);
	return points;
}

cv::Mat CameraModel::GetPixels(const cv::Mat& points, const cv::Mat& camera_matrix, const cv::Mat& depth_map)
{
	// �õ����ͼÿ�����Ӧ��rgbͼ���е�����
	cv::Mat pixels = camera_matrix * points;

	// ��ȡ�����е�Ԫ��
	cv::Mat z = pixels.row(2);

	// ����һ������ÿ��Ԫ�ض��Ƕ�Ӧ�еĵ�����Ԫ�صĵ���
	cv::Mat inverse_z;
	cv::divide(1.0, z, inverse_z);
	cv::Mat inverse_z_extended = cv::repeat(inverse_z, 3, 1); // [1, 480*640] -> [3, 480*640]

	// ��ԭʼ�����뵹��������Ԫ�����
	pixels = pixels.mul(inverse_z_extended);

	// ����һ��������������Ԫ�ض���1
	cv::Mat onesRow = cv::Mat::ones(1, pixels.cols, pixels.type());

	// ��ȡҪ������
	cv::Mat targetRow = pixels.row(2);

	// ʹ�� cv::compare ������Ƿ�ȫ��Ϊ1
	cv::Mat comparisonResult;
	cv::compare(targetRow, onesRow, comparisonResult, cv::CMP_EQ);

	// depth_in_rgb: �洢ת����rgbͼ���µ����ͼ
	cv::Mat depth_in_rgb = cv::Mat::zeros(IMAGE_HEIGHT_480, IMAGE_WIDTH_640, CV_16U);

	cv::Mat map_x = cv::Mat::zeros(depth_map.size(), CV_32FC1);
	cv::Mat map_y = cv::Mat::zeros(depth_map.size(), CV_32FC1);


	// ʹ�� OpenMP ���л����ѭ��
	//#pragma omp parallel for
	for (int y = 0; y < IMAGE_HEIGHT_480; ++y)
	{
		int column__ = y * IMAGE_WIDTH_640;
		for (int x = 0; x < IMAGE_WIDTH_640; ++x)
		{
			// ����ԭʼdepth(x, y)�������ֵӦ����rgb�µ�λ��(x_inrgb, y_inrgb)
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
	//��1��INTER_NEAREST��������ڲ�ֵ
	//��2��INTER_LINEAR����˫���Բ�ֵ��Ĭ�ϣ�
	//��3��INTER_CUBIC����˫��������ֵ����4��4���������ڵ�˫���β�ֵ��
	// (4��INTER_LANCZOS4����lanczos��ֵ����8��8���������Lanczos��ֵ��

	return depth_in_rgb;
}
