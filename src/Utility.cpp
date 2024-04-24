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

//auto Utility::GetFeaturePointsPixels(const std::string& feature_rgb_path, std::vector<std::vector<std::string>>& feature_pixels_position, char delimiter)
//{
//	//// 打开文本文件
//	std::ifstream file(feature_rgb_path);
//
//	// 检查文件是否成功打开
//	if (!file.is_open()) {
//		std::cerr << "Error opening file" << std::endl;
//		//return 1;
//		exit(1);
//	}
//
//
//	std::string line;
//	// 逐行读取文件内容
//	for (int currentRow = 0; std::getline(file, line); ++currentRow) {
//		// 使用字符串流解析每一行的数据
//		std::istringstream iss(line);
//		std::string value;
//		std::vector<std::string> values;
//		//std::vector<std::vector<std::string>> feature_pixels_position;
//		if (currentRow == 0) { continue; }
//
//		int currentColumn = 0;
//		int nextFeature = 0;
//		// 使用字段分隔符拆分每一行的数据
//		while (std::getline(iss, value, delimiter)) {
//			++currentColumn;
//
//			// 如果当前列是目标列，则添加到 vector 中
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
			//std::cout << i << '\n';

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


//void Utility::saveToTxt(const std::vector<std::vector<float>>& data, const std::string& filename, char delimiter)
//{
//	// 打开文件流
//	std::ofstream outfile(filename);
//
//	// 检查文件是否成功打开
//	if (!outfile.is_open()) {
//		std::cerr << "Error opening file: " << filename << std::endl;
//		return;
//	}
//
//	// 遍历二维数组并将数据写入文件
//	for (const auto& row : data) {
//		for (size_t i = 0; i < row.size(); ++i) {
//			outfile << row[i];
//
//			// 在每个元素后面添加分隔符，除了最后一个元素
//			if (i < row.size() - 1) {
//				outfile << delimiter;
//			}
//		}
//		// 换行表示新的一行
//		outfile << std::endl;
//	}
//
//	// 关闭文件流
//	outfile.close();
//}



cv::Mat Utility::PositionToMotion(const cv::Mat& p1, const cv::Mat& p2, const cv::Mat& p3, const cv::Mat& p4)
{
	//std::cout << p1 << p2 << p3 << p4 << '\n';

	cv::Mat p2_p3 = p2 - p3;
	cv::Mat p1_p3 = p1 - p3;

	// n1, n2, n3分别代表面部建立的三个坐标轴的方向
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

void Utility::DrawCoord(cv::Mat& image, const cv::Mat& rot_mat, const cv::Mat& p1, const cv::Mat& p2, const cv::Mat& p3, const cv::Mat& p4)
{
	cv::Mat p2_p3 = p2 - p3;
	cv::Mat p1_p3 = p1 - p3;

	// n1, n2, n3分别代表面部建立的三个坐标轴的方向
	cv::Mat n1 = p2_p3.cross(p1_p3);
	cv::normalize(n1, n1);
	cv::Mat n2 = p2_p3 - p1_p3;
	cv::normalize(n2, n2);
	cv::Mat n3 = n1.cross(n2);

	cv::Mat center_position = p4 - 80.0f * n1;
	float x = center_position.at<float>(0, 0);
	float y = center_position.at<float>(1, 0);
	float z = center_position.at<float>(2, 0);

	cv::Mat axis_1 = center_position - 100 * n1;
	cv::Mat axis_2 = center_position - 100 * n2;
	cv::Mat axis_3 = center_position - 100 * n3;

	cv::Mat center = rot_mat * (center_position / center_position.at<float>(0, 2));
	cv::Mat axis_1_end = rot_mat * (axis_1.clone() / axis_1.at<float>(0,2));
	cv::Mat axis_2_end = rot_mat * (axis_2.clone() / axis_2.at<float>(0, 2));
	cv::Mat axis_3_end = rot_mat * (axis_3.clone() / axis_3.at<float>(0, 2));

	cv::Point center_pixel(static_cast<int>(center.at<float>(0, 0)), static_cast<int>(center.at<float>(0, 1)));
	cv::Point axis_1_pixel(static_cast<int>(axis_1_end.at<float>(0, 0)), static_cast<int>(axis_1_end.at<float>(0, 1)));
	cv::Point axis_2_pixel(static_cast<int>(axis_2_end.at<float>(0, 0)), static_cast<int>(axis_2_end.at<float>(0, 1)));
	cv::Point axis_3_pixel(static_cast<int>(axis_3_end.at<float>(0, 0)), static_cast<int>(axis_3_end.at<float>(0, 1)));

	// 在图像上绘制直线
	cv::line(image, center_pixel, axis_1_pixel, cv::Scalar(0, 0, 255), 2); 
	cv::line(image, center_pixel, axis_2_pixel, cv::Scalar(0, 255, 0), 2); 
	cv::line(image, center_pixel, axis_3_pixel, cv::Scalar(255, 0, 0), 2); 

	//std::cout << "Utility::DrawCoord" << '\n';
	//std::cout << center_pixel << axis_1_pixel << axis_2_pixel << axis_3_pixel << '\n';
	//std::cout << center_position << axis_1 << axis_2 << axis_3 << '\n';
	//std::cout << center << axis_1_end << axis_2_end << axis_3_end << '\n';
	//std::cout << center_position / center_position.at<float>(0, 2) << '\n';
	//std::cout << rot_mat << '\n';



}

void Utility::GetFeaturePointsPixels(const std::string& feature_rgb_path, std::vector<std::vector<std::string>>& feature_pixels_position, char delimiter)
{
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
				//std::cout << currentRow - 1 << " " << currentColumn << " " << value << "\n";
				feature_pixels_position[nextFeature][currentRow - 1] = value;
				nextFeature++;
			}
		}


	}
	return;

}




void Utility::GetFeaturePointsPixels_givenseries(const std::string& feature_rgb_path, std::vector<std::vector<std::string>>& feature_pixels_position, const std::vector<int>& id_series, char delimiter)
{
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
			if (currentColumn == id_series[nextFeature] + 1) {
				//std::cout << currentRow - 1 << " " << currentColumn << " " << value << "\n";
				feature_pixels_position[nextFeature][currentRow - 1] = value;
				nextFeature++;
			}
		}


	}
	return;

}


