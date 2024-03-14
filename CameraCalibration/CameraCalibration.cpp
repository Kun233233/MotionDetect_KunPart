#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

// 定义棋盘的尺寸
//int CHECKERBOARD[2]{ 9,12 };
int CHECKERBOARD[2]{ 8,11 };
//int CHECKERBOARD[2]{ 7,10 };
//int CHECKERBOARD[2]{ 6,8 };
//double squaresize = 12;
double squaresize = 10;

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
    //std::string path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/Depth/*.png";
    std::string path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/RGB/*.png";

    // 存储成图片形式的地址
    //std::string  drawn_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/Depth_drawn";
    std::string  drawn_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/RGB_drawn";

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

    return 0;
}

