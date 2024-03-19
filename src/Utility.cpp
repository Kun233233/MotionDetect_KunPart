#include "Utility.h"

float Utility::PI = 3.1415926535f;

Utility::Utility()
{
	//PI = 3.1415926535f;
}

Utility::~Utility()
{
}

std::string Utility::getValueAt(std::ifstream& file, int targetRow, int targetColumn, char delimiter)
{
	std::string line;

	// ���ж�ȡ�ļ�����
	for (int currentRow = 1; std::getline(file, line); ++currentRow) {
		if (currentRow == targetRow) {
			// ʹ���ַ���������ÿһ�е�����
			std::istringstream iss(line);
			std::string value;
			std::vector<std::string> values;

			// ʹ���ֶηָ������ÿһ�е�����
			while (std::getline(iss, value, delimiter)) {
				values.push_back(value);
			}

			// ���Ŀ�����Ƿ�����Ч��Χ��
			if (targetColumn >= 1 && targetColumn <= values.size()) {
				return values[targetColumn - 1]; // ����Ŀ���е�ֵ
			}
			else {
				std::cerr << "Error: Column " << targetColumn << " out of range." << std::endl;
				return "";
			}
		}
	}

	// ����ļ���δ�ҵ�Ŀ����
	std::cerr << "Error: Row " << targetRow << " not found." << std::endl;
	return "";
}

//auto Utility::GetFeaturePointsPixels(const std::string& feature_rgb_path, std::vector<std::vector<std::string>>& feature_pixels_position, char delimiter)
//{
//	//// ���ı��ļ�
//	std::ifstream file(feature_rgb_path);
//
//	// ����ļ��Ƿ�ɹ���
//	if (!file.is_open()) {
//		std::cerr << "Error opening file" << std::endl;
//		//return 1;
//		exit(1);
//	}
//
//
//	std::string line;
//	// ���ж�ȡ�ļ�����
//	for (int currentRow = 0; std::getline(file, line); ++currentRow) {
//		// ʹ���ַ���������ÿһ�е�����
//		std::istringstream iss(line);
//		std::string value;
//		std::vector<std::string> values;
//		//std::vector<std::vector<std::string>> feature_pixels_position;
//		if (currentRow == 0) { continue; }
//
//		int currentColumn = 0;
//		int nextFeature = 0;
//		// ʹ���ֶηָ������ÿһ�е�����
//		while (std::getline(iss, value, delimiter)) {
//			++currentColumn;
//
//			// �����ǰ����Ŀ���У�����ӵ� vector ��
//			if (currentColumn == feature_id_series[nextFeature] + 1) {
//				std::cout << currentRow - 1 << " " << currentColumn << " " << value << "\n";
//				feature_pixels_position[nextFeature][currentRow - 1] = value;
//				nextFeature++;
//			}
//		}
//
//
//	}
//	return;
//
//}

void Utility::saveToTxt(const std::vector<std::vector<std::string>>& data, const std::string& filename, char delimiter)
{
	// ���ļ���
	std::ofstream outfile(filename);

	// ����ļ��Ƿ�ɹ���
	if (!outfile.is_open()) {
		std::cerr << "Error opening file: " << filename << std::endl;
		return;
	}

	// ������ά���鲢������д���ļ�
	for (const auto& row : data) {
		for (size_t i = 0; i < row.size(); ++i) {
			outfile << row[i];
			//std::cout << i << '\n';

			// ��ÿ��Ԫ�غ�����ӷָ������������һ��Ԫ��
			if (i < row.size() - 1) {
				outfile << delimiter;
			}
		}
		// ���б�ʾ�µ�һ��
		outfile << std::endl;
	}

	// �ر��ļ���
	outfile.close();
}


//void Utility::saveToTxt(const std::vector<std::vector<float>>& data, const std::string& filename, char delimiter)
//{
//	// ���ļ���
//	std::ofstream outfile(filename);
//
//	// ����ļ��Ƿ�ɹ���
//	if (!outfile.is_open()) {
//		std::cerr << "Error opening file: " << filename << std::endl;
//		return;
//	}
//
//	// ������ά���鲢������д���ļ�
//	for (const auto& row : data) {
//		for (size_t i = 0; i < row.size(); ++i) {
//			outfile << row[i];
//
//			// ��ÿ��Ԫ�غ�����ӷָ������������һ��Ԫ��
//			if (i < row.size() - 1) {
//				outfile << delimiter;
//			}
//		}
//		// ���б�ʾ�µ�һ��
//		outfile << std::endl;
//	}
//
//	// �ر��ļ���
//	outfile.close();
//}



cv::Mat Utility::PositionToMotion(const cv::Mat& p1, const cv::Mat& p2, const cv::Mat& p3, const cv::Mat& p4)
{
	//std::cout << p1 << p2 << p3 << p4 << '\n';

	cv::Mat p2_p3 = p2 - p3;
	cv::Mat p1_p3 = p1 - p3;

	// n1, n2, n3�ֱ�����沿����������������ķ���
	cv::Mat n1 = p2_p3.cross(p1_p3);
	cv::normalize(n1, n1);
	cv::Mat n2 = p2_p3 - p1_p3;
	cv::normalize(n2, n2);
	cv::Mat n3 = n1.cross(n2);
	//std::cout << n1 << '\n' << n2 << '\n' << n3 << '\n' << '\n';
	//std::cout << n2.at<float>(2, 0) << '\n' << n3.at<float>(2, 0) << '\n' << n2.at<float>(2, 0) / n3.at<float>(2, 0) << '\n' << std::asinf(n2.at<float>(2, 0) / n3.at<float>(2, 0)) << '\n' << '\n';
	float a = std::atan2f(n2.at<float>(2, 0), n3.at<float>(2, 0));
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
