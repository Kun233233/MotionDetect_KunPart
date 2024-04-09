#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <fstream>

using namespace std;
using namespace cv;
// 定义棋盘的尺寸
//int CHECKERBOARD[2]{ 9,12 };
int CHECKERBOARD[2]{ 8,11 };
//int CHECKERBOARD[2]{ 7,10 };
//int CHECKERBOARD[2]{ 6,8 };
//double squaresize = 12;
double squaresize = 10;

void initUndistAndRemap(vector<Mat> imgs, Mat cameraMatrix, Mat distCoeffs, Size imageSize, vector<Mat>& undistImgs)
{
    // 计算映射坐标矩阵
    Mat R = Mat::eye(3, 3, CV_32F);
    Mat mapx = Mat(imageSize, CV_32FC1);
    Mat mapy = Mat(imageSize, CV_32FC1);
    // 内参矩阵/畸变系数/...
    initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, imageSize, CV_32FC1, mapx, mapy);
    for (int i = 0; i < imgs.size(); i++)
    {
        Mat undistImg;
        remap(imgs[i], undistImg, mapx, mapy, INTER_LINEAR);
        undistImgs.push_back(undistImg);
    }
}




int main()
{
    // 创建向量以存储每个棋盘图像的 3D 点向量
    std::vector<std::vector<cv::Point3f> > objpoints;

    // 创建向量以存储每个棋盘图像的 2D 点向量
    std::vector<std::vector<cv::Point2f> > imgpoints;

    // 定义 3D 点的世界坐标
    std::vector<cv::Point3f> objp;
    for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
    {
        for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j * squaresize, i * squaresize, 0));
    }


    // 提取存储在给定目录中的单个图像的路径
    std::vector<cv::String> images;
    // 包含棋盘图像的文件夹的路径
    std::string path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/Depth/*.png";
    //std::string path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/RGB/*.png";

    // 存储成图片形式的地址
    std::string  drawn_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/Depth_drawn";
    //std::string  drawn_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/RGB_drawn";

    cv::glob(path, images);

    cv::Mat frame, gray, gray_filtered;
    // 用于存储检测到的棋盘角的像素坐标的向量
    std::vector<cv::Point2f> corner_pts;
    bool success;

    /* 循环遍历目录中的所有图像 */
    for (int i{ 0 }; i < images.size(); i++)
    {
        std::cout << i << "\n";
        frame = cv::imread(images[i]);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::bilateralFilter(gray, gray_filtered, 15, 60, 60);
        //gray = gray_filtered;

        // 寻找棋盘角
        // 如果在图像中找到所需数量的角，则成功 = true
        success = cv::findChessboardCorners(gray_filtered, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts);

        /*
         如果检测到所需数量的角，们细化像素坐标并在棋盘格图像上显示它们
        */
        if (success)
        {
            //cv::TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

            // 细化给定二维点的像素坐标。
            cv::cornerSubPix(gray_filtered, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            // 在棋盘上显示检测到的角点
            cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
            //cv::drawChessboardCorners(gray_filtered, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }

        cv::imshow("Image", frame);
        std::string drawn_name = drawn_folder_path + "/" + std::to_string(i) + ".png";
        cv::imwrite(drawn_name, frame);
        cv::waitKey(1000);
    }

    cv::destroyAllWindows();

    cv::Mat cameraMatrix, distCoeffs, R, T;

    /*通过传递已知 3D 点 (objpoints) 的值 和检测到的角点（imgpoints）对应的像素坐标 实现相机标定*/
    cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);

    std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
    std::cout << "distCoeffs : " << distCoeffs << std::endl;
    std::cout << "Rotation vector : " << R << std::endl;
    std::cout << "Translation vector : " << T << std::endl;


    //// -6 图像去畸变
    //std::vector<cv::Mat> undistImgs;
    //// 使用initUndistortRectifyMap()函数和remap()函数校正图像
    //initUndistAndRemap(imgs, cameraMatrix, distCoeffs, imageSize, undistImgs); // 畸变图像/前文计算得到的内参矩阵/畸变系数/图像尺寸/去畸变后的图像,自定义函数是为了处理多副图像
    //// undist(imgs, cameraMatrix, distCoeffs, undistImgs);//用undistort()函数直接计算校正图像,自定义函数是为了处理多副图像
    //// 显示校正前后的图像(一张示例)
    //for (int i = 0; i < 1; i++)
    //{
    //    string windowNumber = to_string(i);
    //    imshow("chessboard corners without undistort -- image" + windowNumber, imgs[i]);
    //    imshow("chessboard corners with undistort -- image" + windowNumber, undistImgs[i]);
    //    imwrite("../chessboard corners without undistort.png", imgs[i]);
    //    imwrite("../chessboard corners with undistort.png", undistImgs[i]);
    //}
    //waitKey(0);

    //// -7 单目投影(重投影)：根据成像模型及空间点三位坐标计算图像二维坐标
    //vector<vector<Point2f>> imagePoints; // 存放二维坐标
    //// 根据三维坐标和相机与世界坐标系时间的关系估计内角点像素坐标
    //for (int i = 0; i < imgs.size(); i++)
    //{
    //    Mat rvec = R[i], tvec = T[i];
    //    vector<Point3f> PointSets = objpoints[i];
    //    vector<Point2f> imagePoint;                                                 // 存放二维坐标
    //    projectPoints(PointSets, rvec, tvec, cameraMatrix, distCoeffs, imagePoint); // 输入三维点/旋转及平移向量/前文计算得到的内参矩阵和畸变矩阵/输出二维点
    //    imagePoints.push_back(imagePoint);
    //}

    //// -8 计算重投影误差
    //vector<vector<double>> ReProjectionError;
    //vector<vector<double>> ReProjectionErrorX;
    //vector<vector<double>> ReProjectionErrorY;
    //double e = 0.0;
    //for (int i = 0; i < imgs.size(); i++)
    //{
    //    vector<Point2f> imagePoint = imagePoints[i]; // 存放二维坐标
    //    vector<double> er;
    //    vector<double> erX;
    //    vector<double> erY;
    //    for (int j = 0; j < imagePoint.size(); j++)
    //    {
    //        double eX = imagePoint[j].x - imgpoints[i][j].x;
    //        double eY = imagePoint[j].y - imgpoints[i][j].y;
    //        double error = sqrt(pow(eX, 2) + pow(eY, 2));
    //        erX.push_back(eX);
    //        erY.push_back(eY);
    //        er.push_back(error);
    //        e += error;
    //    }
    //    ReProjectionError.push_back(er);
    //    ReProjectionErrorX.push_back(erX);
    //    ReProjectionErrorY.push_back(erY);
    //}
    //// 计算估计值和图像中计算的真实时之间的平均误差
    //cout << "平均重投影误差:" << e / (imagePoints[0].size() * imgs.size()) << endl;

    //// -9 保存重投影误差数据
    //ofstream ReProjectionErrorFile;
    //ReProjectionErrorFile.open("../ReProjectionError.txt");
    //ofstream ReProjectionErrorFileX;
    //ReProjectionErrorFileX.open("../ReProjectionErrorX.txt");
    //ofstream ReProjectionErrorFileY;
    //ReProjectionErrorFileY.open("../ReProjectionErrorY.txt");
    //if (!ReProjectionErrorFile.is_open() || !ReProjectionErrorFileX.is_open() || !ReProjectionErrorFileY.is_open())
    //{
    //    exit(EXIT_FAILURE);
    //}
    //ReProjectionErrorFile << fixed;
    //ReProjectionErrorFile.precision(5);
    //ReProjectionErrorFileX << fixed;
    //ReProjectionErrorFileX.precision(5);
    //ReProjectionErrorFileY << fixed;
    //ReProjectionErrorFileY.precision(5);
    //for (int i = 0; i < imgs.size(); i++)
    //{
    //    for (int j = 0; j < imagePoints[i].size(); j++)
    //    {
    //        ReProjectionErrorFile << ReProjectionError[i][j] << " ";
    //        ReProjectionErrorFileX << ReProjectionErrorX[i][j] << " ";
    //        ReProjectionErrorFileY << ReProjectionErrorY[i][j] << " ";
    //    }
    //    ReProjectionErrorFile << endl;
    //    ReProjectionErrorFileX << endl;
    //    ReProjectionErrorFileY << endl;
    //}
    //ReProjectionErrorFile.close();
    //ReProjectionErrorFileX.close();
    //ReProjectionErrorFileY.close();


    return 0;
}




//#include <opencv2/opencv.hpp>
//#include <fstream>
//#include <iostream>
//#include <vector>
//#include <cstring>
//#include <sstream>
//#include <cstdlib>
//using namespace std;
//using namespace cv;
//
//void initUndistAndRemap(vector<Mat> imgs, Mat cameraMatrix, Mat distCoeffs, Size imageSize, vector<Mat>& undistImgs)
//{
//    // 计算映射坐标矩阵
//    Mat R = Mat::eye(3, 3, CV_32F);
//    Mat mapx = Mat(imageSize, CV_32FC1);
//    Mat mapy = Mat(imageSize, CV_32FC1);
//    // 内参矩阵/畸变系数/...
//    initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, imageSize, CV_32FC1, mapx, mapy);
//    for (int i = 0; i < imgs.size(); i++)
//    {
//        Mat undistImg;
//        remap(imgs[i], undistImg, mapx, mapy, INTER_LINEAR);
//        undistImgs.push_back(undistImg);
//    }
//}
//
//void undist(vector<Mat> imgs, Mat cameraMatrix, Mat distCoeffs, vector<Mat>& undistImgs)
//{
//    for (int i = 0; i < imgs.size(); i++)
//    {
//        Mat undistImg;
//        // 单幅图像去畸变:畸变图像/去畸变后的图像/内参矩阵/畸变系数
//        undistort(imgs[i], undistImg, cameraMatrix, distCoeffs);
//        undistImgs.push_back(undistImg);
//    }
//}
//
//bool LoadData(const string& imagePath, const string& imageFilename, vector<Mat>& imgs)
//{
//    ifstream Left;
//    Left.open(imageFilename.c_str());
//    while (!Left.eof())
//    {
//        string s;
//        getline(Left, s);
//        if (!s.empty())
//        {
//            stringstream ss;
//            ss << s;
//            double t;
//            string imageName;
//            ss >> t;
//            ss >> imageName;
//            Mat img = imread(imagePath + "/" + imageName);
//            imgs.push_back(img);
//            if (!img.data)
//            {
//                cout << "请输入正确的图像文件" << endl;
//                return 0;
//            }
//        }
//    }
//
//    return 1;
//}
//
//int main(int argc, char** argv)
//{
//    if (argc != 6)
//    {
//        cerr << endl
//            << "Usage: ./CameraCalibration path_to_CalibrateImage path_to_calibdata.txt board_size_cols board_size_rows corners_of_checkerboard(mm)" << endl
//            << "eg: ./CameraCalibration ../CalibrateData ../CalibrateData/calibdata.txt 9 6 10" << endl;
//
//        return 1;
//    }
//
//    // -1 读取数据
//    vector<Mat> imgs;
//    LoadData(argv[1], argv[2], imgs);
//    // 棋盘格内角点行列数
//    int bcols, brows;
//    // 棋盘格每个方格的真实尺寸
//    double side_length;
//    stringstream ss;
//    ss << argv[3];
//    ss >> bcols;
//    ss.clear();
//    ss << argv[4];
//    ss >> brows;
//    ss.clear();
//    ss << argv[5];
//    ss >> side_length;
//    Size board_size = Size(bcols, brows);
//
//    // -2 提取并计算标定板角点像素坐标
//    // 多副图像分别放入vector<vector<Point2f>>
//    vector<vector<Point2f>> imgsPoints;
//    for (int i = 0; i < imgs.size(); i++)
//    {
//        Mat img1 = imgs[i], gray1;
//        cvtColor(img1, gray1, COLOR_BGR2GRAY);
//        vector<Point2f> img1_points;
//        // 第一个参数是输入的棋盘格图像
//        // 第二个参数是棋盘格内部的角点的行列数（注意：不是棋盘格的行列数，而内部角点-不包括边缘-的行列数）
//        // 第三个参数是检测到的棋盘格角点，类型为std::vector<cv::Point2f>
//        bool ret = findChessboardCorners(gray1, board_size, img1_points);
//
//        // 细化标定板角点坐标（亚像素）,Size(5, 5)为细化方格坐标领域范围
//        find4QuadCornerSubpix(gray1, img1_points, Size(5, 5));
//
//        // 第一个参数是棋盘格图像（8UC3）既是输入也是输出
//        // 第二个参数是棋盘格内部角点的行、列
//        // 第三个参数是检测到的棋盘格角点
//        // 第四个参数是cv::findChessboardCorners()的返回值。
//        drawChessboardCorners(img1, board_size, img1_points, ret);
//        // string windowNumber = to_string(i);
//        // imshow("chessboard corners"+windowNumber, img1);
//
//        imgsPoints.push_back(img1_points);
//    }
//
//    // -3 使用棋盘格每个内角点的世界坐标
//    vector<vector<Point3f>> objectPoints; // 空间三维坐标（位于一个平面内，以此为xy坐标平面）
//    for (int i = 0; i < imgsPoints.size(); i++)
//    {
//        vector<Point3f> tempPointSet;
//        for (int j = 0; j < board_size.height; j++)
//        {
//            for (int k = 0; k < board_size.width; k++)
//            {
//                // 假设标定板为世界坐标系的z平面，即z=0
//                Point3f realPoint;
//                realPoint.x = j * side_length;
//                realPoint.y = k * side_length;
//                realPoint.z = 0;
//                tempPointSet.push_back(realPoint);
//            }
//        }
//        objectPoints.push_back(tempPointSet);
//    }
//
//    // -4 内参及畸变标定
//    Size imageSize; // 图像尺寸
//    imageSize.width = imgs[0].cols;
//    imageSize.height = imgs[0].rows;
//    // 定义内外参
//    Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 相机内参数矩阵
//    Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));   // 相机的5个畸变系数
//    vector<Mat> rvecs, tvecs;                               // 每幅图像的旋转向量/平移向量
//    // 调用OpenCV标定函数
//    calibrateCamera(objectPoints, imgsPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, 0);
//    cout << "相机的内参矩阵K=" << endl
//        << cameraMatrix << endl;
//    cout << "相机畸变系数:" << endl
//        << distCoeffs << endl;
//
//    // -5 保存数据
//    ofstream calibrateFile;
//    // 写入数据:
//    calibrateFile.open("../calibrateCamera.txt"); // 没有自动创建
//    if (!calibrateFile.is_open())                 // 文件是否打开
//    {
//        exit(EXIT_FAILURE); // 终止程序
//    }
//    calibrateFile << fixed;     // 开始写入,与cout相同
//    calibrateFile.precision(5); // 写入小数精度
//    calibrateFile << "cameraMatrix K=" << endl
//        << cameraMatrix; // 写入数据（覆盖
//    calibrateFile << endl << "distCoeffs=" << endl
//        << distCoeffs;
//    calibrateFile.close();
//
//    // -6 图像去畸变
//    vector<Mat> undistImgs;
//    // 使用initUndistortRectifyMap()函数和remap()函数校正图像
//    initUndistAndRemap(imgs, cameraMatrix, distCoeffs, imageSize, undistImgs); // 畸变图像/前文计算得到的内参矩阵/畸变系数/图像尺寸/去畸变后的图像,自定义函数是为了处理多副图像
//    // undist(imgs, cameraMatrix, distCoeffs, undistImgs);//用undistort()函数直接计算校正图像,自定义函数是为了处理多副图像
//    // 显示校正前后的图像(一张示例)
//    for (int i = 0; i < 1; i++)
//    {
//        string windowNumber = to_string(i);
//        imshow("chessboard corners without undistort -- image" + windowNumber, imgs[i]);
//        imshow("chessboard corners with undistort -- image" + windowNumber, undistImgs[i]);
//        imwrite("../chessboard corners without undistort.png", imgs[i]);
//        imwrite("../chessboard corners with undistort.png", undistImgs[i]);
//    }
//    waitKey(0);
//
//    // -7 单目投影(重投影)：根据成像模型及空间点三位坐标计算图像二维坐标
//    vector<vector<Point2f>> imagePoints; // 存放二维坐标
//    // 根据三维坐标和相机与世界坐标系时间的关系估计内角点像素坐标
//    for (int i = 0; i < imgs.size(); i++)
//    {
//        Mat rvec = rvecs[i], tvec = tvecs[i];
//        vector<Point3f> PointSets = objectPoints[i];
//        vector<Point2f> imagePoint;                                                 // 存放二维坐标
//        projectPoints(PointSets, rvec, tvec, cameraMatrix, distCoeffs, imagePoint); // 输入三维点/旋转及平移向量/前文计算得到的内参矩阵和畸变矩阵/输出二维点
//        imagePoints.push_back(imagePoint);
//    }
//
//    // -8 计算重投影误差
//    vector<vector<double>> ReProjectionError;
//    vector<vector<double>> ReProjectionErrorX;
//    vector<vector<double>> ReProjectionErrorY;
//    double e = 0.0;
//    for (int i = 0; i < imgs.size(); i++)
//    {
//        vector<Point2f> imagePoint = imagePoints[i]; // 存放二维坐标
//        vector<double> er;
//        vector<double> erX;
//        vector<double> erY;
//        for (int j = 0; j < imagePoint.size(); j++)
//        {
//            double eX = imagePoint[j].x - imgsPoints[i][j].x;
//            double eY = imagePoint[j].y - imgsPoints[i][j].y;
//            double error = sqrt(pow(eX, 2) + pow(eY, 2));
//            erX.push_back(eX);
//            erY.push_back(eY);
//            er.push_back(error);
//            e += error;
//        }
//        ReProjectionError.push_back(er);
//        ReProjectionErrorX.push_back(erX);
//        ReProjectionErrorY.push_back(erY);
//    }
//    // 计算估计值和图像中计算的真实时之间的平均误差
//    cout << "平均重投影误差:" << e / (imagePoints[0].size() * imgs.size()) << endl;
//
//    // -9 保存重投影误差数据
//    ofstream ReProjectionErrorFile;
//    ReProjectionErrorFile.open("../ReProjectionError.txt");
//    ofstream ReProjectionErrorFileX;
//    ReProjectionErrorFileX.open("../ReProjectionErrorX.txt");
//    ofstream ReProjectionErrorFileY;
//    ReProjectionErrorFileY.open("../ReProjectionErrorY.txt");
//    if (!ReProjectionErrorFile.is_open() || !ReProjectionErrorFileX.is_open() || !ReProjectionErrorFileY.is_open())
//    {
//        exit(EXIT_FAILURE);
//    }
//    ReProjectionErrorFile << fixed;
//    ReProjectionErrorFile.precision(5);
//    ReProjectionErrorFileX << fixed;
//    ReProjectionErrorFileX.precision(5);
//    ReProjectionErrorFileY << fixed;
//    ReProjectionErrorFileY.precision(5);
//    for (int i = 0; i < imgs.size(); i++)
//    {
//        for (int j = 0; j < imagePoints[i].size(); j++)
//        {
//            ReProjectionErrorFile << ReProjectionError[i][j] << " ";
//            ReProjectionErrorFileX << ReProjectionErrorX[i][j] << " ";
//            ReProjectionErrorFileY << ReProjectionErrorY[i][j] << " ";
//        }
//        ReProjectionErrorFile << endl;
//        ReProjectionErrorFileX << endl;
//        ReProjectionErrorFileY << endl;
//    }
//    ReProjectionErrorFile.close();
//    ReProjectionErrorFileX.close();
//    ReProjectionErrorFileY.close();
//
//    /*
//    // 圆形标定板角点提取
//    Mat img2 = imread("../CalibrateData/circle.png");
//    Mat gray2;
//    cvtColor(img2, gray2, COLOR_BGR2GRAY);
//    Size board_size2 = Size(7, 7);          //圆形标定板圆心数目（行，列）
//    vector<Point2f> img2_points;//检测角点，单副图像放入vector<Point2f>
//    findCirclesGrid(gray2, board_size2, img2_points);           //计算圆形标定板检点
//    find4QuadCornerSubpix(gray2, img2_points, Size(5, 5));      //细化圆形标定板角点坐标
//    drawChessboardCorners(img2, board_size2, img2_points, true);
//    imshow("圆形标定板角点检测结果", img2);
//    waitKey();
//    */
//
//    return 0;
//}

