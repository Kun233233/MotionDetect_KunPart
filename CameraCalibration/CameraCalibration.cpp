#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

// �������̵ĳߴ�
//int CHECKERBOARD[2]{ 9,12 };
//int CHECKERBOARD[2]{ 8,11 };
//int CHECKERBOARD[2]{ 7,10 };
int CHECKERBOARD[2]{ 6,8 };
//double squaresize = 12;
double squaresize = 1;

int main()
{
    // ���������Դ洢ÿ������ͼ��� 3D ������
    std::vector<std::vector<cv::Point3f> > objpoints;

    // ���������Դ洢ÿ������ͼ��� 2D ������
    std::vector<std::vector<cv::Point2f> > imgpoints;

    // ���� 3D �����������
    std::vector<cv::Point3f> objp;
    for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
    {
        for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j * squaresize, i * squaresize, 0));
    }


    // ��ȡ�洢�ڸ���Ŀ¼�еĵ���ͼ���·��
    std::vector<cv::String> images;
    // ��������ͼ����ļ��е�·��
    //std::string path = "D:/aaaLab/aaagraduate/SaveVideo/src/Calibration/Depth/*.png";
    std::string path = "D:/aaaLab/aaagraduate/SaveVideo/src/Calibration/RGB/*.png";

    cv::glob(path, images);

    cv::Mat frame, gray, gray_filtered;
    // ���ڴ洢��⵽�����̽ǵ��������������
    std::vector<cv::Point2f> corner_pts;
    bool success;

    /* ѭ������Ŀ¼�е�����ͼ�� */
    for (int i{ 0 }; i < images.size(); i++)
    {
        std::cout << i << "\n";
        frame = cv::imread(images[i]);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::bilateralFilter(gray, gray_filtered, 15, 60, 60);
        //gray = gray_filtered;

        // Ѱ�����̽�
        // �����ͼ�����ҵ����������Ľǣ���ɹ� = true
        success = cv::findChessboardCorners(gray_filtered, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts);

        /*
         �����⵽���������Ľǣ���ϸ���������겢�����̸�ͼ������ʾ����
        */
        if (success)
        {
            //cv::TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

            // ϸ��������ά����������ꡣ
            cv::cornerSubPix(gray_filtered, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            // ����������ʾ��⵽�Ľǵ�
            cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
            //cv::drawChessboardCorners(gray_filtered, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }

        cv::imshow("Image", frame);
        cv::waitKey(1000);
    }

    cv::destroyAllWindows();

    cv::Mat cameraMatrix, distCoeffs, R, T;

    /*ͨ��������֪ 3D �� (objpoints) ��ֵ �ͼ�⵽�Ľǵ㣨imgpoints����Ӧ���������� ʵ������궨*/
    cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);

    std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
    std::cout << "distCoeffs : " << distCoeffs << std::endl;
    std::cout << "Rotation vector : " << R << std::endl;
    std::cout << "Translation vector : " << T << std::endl;

    return 0;
}

