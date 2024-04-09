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
// �������̵ĳߴ�
//int CHECKERBOARD[2]{ 9,12 };
int CHECKERBOARD[2]{ 8,11 };
//int CHECKERBOARD[2]{ 7,10 };
//int CHECKERBOARD[2]{ 6,8 };
//double squaresize = 12;
double squaresize = 10;

void initUndistAndRemap(vector<Mat> imgs, Mat cameraMatrix, Mat distCoeffs, Size imageSize, vector<Mat>& undistImgs)
{
    // ����ӳ���������
    Mat R = Mat::eye(3, 3, CV_32F);
    Mat mapx = Mat(imageSize, CV_32FC1);
    Mat mapy = Mat(imageSize, CV_32FC1);
    // �ڲξ���/����ϵ��/...
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
    std::string path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/Depth/*.png";
    //std::string path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/RGB/*.png";

    // �洢��ͼƬ��ʽ�ĵ�ַ
    std::string  drawn_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/Depth_drawn";
    //std::string  drawn_folder_path = "D:/aaaLab/aaagraduate/SaveVideo/source/Calibration/RGB_drawn";

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
        std::string drawn_name = drawn_folder_path + "/" + std::to_string(i) + ".png";
        cv::imwrite(drawn_name, frame);
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


    //// -6 ͼ��ȥ����
    //std::vector<cv::Mat> undistImgs;
    //// ʹ��initUndistortRectifyMap()������remap()����У��ͼ��
    //initUndistAndRemap(imgs, cameraMatrix, distCoeffs, imageSize, undistImgs); // ����ͼ��/ǰ�ļ���õ����ڲξ���/����ϵ��/ͼ��ߴ�/ȥ������ͼ��,�Զ��庯����Ϊ�˴���ัͼ��
    //// undist(imgs, cameraMatrix, distCoeffs, undistImgs);//��undistort()����ֱ�Ӽ���У��ͼ��,�Զ��庯����Ϊ�˴���ัͼ��
    //// ��ʾУ��ǰ���ͼ��(һ��ʾ��)
    //for (int i = 0; i < 1; i++)
    //{
    //    string windowNumber = to_string(i);
    //    imshow("chessboard corners without undistort -- image" + windowNumber, imgs[i]);
    //    imshow("chessboard corners with undistort -- image" + windowNumber, undistImgs[i]);
    //    imwrite("../chessboard corners without undistort.png", imgs[i]);
    //    imwrite("../chessboard corners with undistort.png", undistImgs[i]);
    //}
    //waitKey(0);

    //// -7 ��ĿͶӰ(��ͶӰ)�����ݳ���ģ�ͼ��ռ����λ�������ͼ���ά����
    //vector<vector<Point2f>> imagePoints; // ��Ŷ�ά����
    //// ������ά������������������ϵʱ��Ĺ�ϵ�����ڽǵ���������
    //for (int i = 0; i < imgs.size(); i++)
    //{
    //    Mat rvec = R[i], tvec = T[i];
    //    vector<Point3f> PointSets = objpoints[i];
    //    vector<Point2f> imagePoint;                                                 // ��Ŷ�ά����
    //    projectPoints(PointSets, rvec, tvec, cameraMatrix, distCoeffs, imagePoint); // ������ά��/��ת��ƽ������/ǰ�ļ���õ����ڲξ���ͻ������/�����ά��
    //    imagePoints.push_back(imagePoint);
    //}

    //// -8 ������ͶӰ���
    //vector<vector<double>> ReProjectionError;
    //vector<vector<double>> ReProjectionErrorX;
    //vector<vector<double>> ReProjectionErrorY;
    //double e = 0.0;
    //for (int i = 0; i < imgs.size(); i++)
    //{
    //    vector<Point2f> imagePoint = imagePoints[i]; // ��Ŷ�ά����
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
    //// �������ֵ��ͼ���м������ʵʱ֮���ƽ�����
    //cout << "ƽ����ͶӰ���:" << e / (imagePoints[0].size() * imgs.size()) << endl;

    //// -9 ������ͶӰ�������
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
//    // ����ӳ���������
//    Mat R = Mat::eye(3, 3, CV_32F);
//    Mat mapx = Mat(imageSize, CV_32FC1);
//    Mat mapy = Mat(imageSize, CV_32FC1);
//    // �ڲξ���/����ϵ��/...
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
//        // ����ͼ��ȥ����:����ͼ��/ȥ������ͼ��/�ڲξ���/����ϵ��
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
//                cout << "��������ȷ��ͼ���ļ�" << endl;
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
//    // -1 ��ȡ����
//    vector<Mat> imgs;
//    LoadData(argv[1], argv[2], imgs);
//    // ���̸��ڽǵ�������
//    int bcols, brows;
//    // ���̸�ÿ���������ʵ�ߴ�
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
//    // -2 ��ȡ������궨��ǵ���������
//    // �ัͼ��ֱ����vector<vector<Point2f>>
//    vector<vector<Point2f>> imgsPoints;
//    for (int i = 0; i < imgs.size(); i++)
//    {
//        Mat img1 = imgs[i], gray1;
//        cvtColor(img1, gray1, COLOR_BGR2GRAY);
//        vector<Point2f> img1_points;
//        // ��һ����������������̸�ͼ��
//        // �ڶ������������̸��ڲ��Ľǵ����������ע�⣺�������̸�������������ڲ��ǵ�-��������Ե-����������
//        // �����������Ǽ�⵽�����̸�ǵ㣬����Ϊstd::vector<cv::Point2f>
//        bool ret = findChessboardCorners(gray1, board_size, img1_points);
//
//        // ϸ���궨��ǵ����꣨�����أ�,Size(5, 5)Ϊϸ��������������Χ
//        find4QuadCornerSubpix(gray1, img1_points, Size(5, 5));
//
//        // ��һ�����������̸�ͼ��8UC3����������Ҳ�����
//        // �ڶ������������̸��ڲ��ǵ���С���
//        // �����������Ǽ�⵽�����̸�ǵ�
//        // ���ĸ�������cv::findChessboardCorners()�ķ���ֵ��
//        drawChessboardCorners(img1, board_size, img1_points, ret);
//        // string windowNumber = to_string(i);
//        // imshow("chessboard corners"+windowNumber, img1);
//
//        imgsPoints.push_back(img1_points);
//    }
//
//    // -3 ʹ�����̸�ÿ���ڽǵ����������
//    vector<vector<Point3f>> objectPoints; // �ռ���ά���꣨λ��һ��ƽ���ڣ��Դ�Ϊxy����ƽ�棩
//    for (int i = 0; i < imgsPoints.size(); i++)
//    {
//        vector<Point3f> tempPointSet;
//        for (int j = 0; j < board_size.height; j++)
//        {
//            for (int k = 0; k < board_size.width; k++)
//            {
//                // ����궨��Ϊ��������ϵ��zƽ�棬��z=0
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
//    // -4 �ڲμ�����궨
//    Size imageSize; // ͼ��ߴ�
//    imageSize.width = imgs[0].cols;
//    imageSize.height = imgs[0].rows;
//    // ���������
//    Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // ����ڲ�������
//    Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));   // �����5������ϵ��
//    vector<Mat> rvecs, tvecs;                               // ÿ��ͼ�����ת����/ƽ������
//    // ����OpenCV�궨����
//    calibrateCamera(objectPoints, imgsPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, 0);
//    cout << "������ڲξ���K=" << endl
//        << cameraMatrix << endl;
//    cout << "�������ϵ��:" << endl
//        << distCoeffs << endl;
//
//    // -5 ��������
//    ofstream calibrateFile;
//    // д������:
//    calibrateFile.open("../calibrateCamera.txt"); // û���Զ�����
//    if (!calibrateFile.is_open())                 // �ļ��Ƿ��
//    {
//        exit(EXIT_FAILURE); // ��ֹ����
//    }
//    calibrateFile << fixed;     // ��ʼд��,��cout��ͬ
//    calibrateFile.precision(5); // д��С������
//    calibrateFile << "cameraMatrix K=" << endl
//        << cameraMatrix; // д�����ݣ�����
//    calibrateFile << endl << "distCoeffs=" << endl
//        << distCoeffs;
//    calibrateFile.close();
//
//    // -6 ͼ��ȥ����
//    vector<Mat> undistImgs;
//    // ʹ��initUndistortRectifyMap()������remap()����У��ͼ��
//    initUndistAndRemap(imgs, cameraMatrix, distCoeffs, imageSize, undistImgs); // ����ͼ��/ǰ�ļ���õ����ڲξ���/����ϵ��/ͼ��ߴ�/ȥ������ͼ��,�Զ��庯����Ϊ�˴���ัͼ��
//    // undist(imgs, cameraMatrix, distCoeffs, undistImgs);//��undistort()����ֱ�Ӽ���У��ͼ��,�Զ��庯����Ϊ�˴���ัͼ��
//    // ��ʾУ��ǰ���ͼ��(һ��ʾ��)
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
//    // -7 ��ĿͶӰ(��ͶӰ)�����ݳ���ģ�ͼ��ռ����λ�������ͼ���ά����
//    vector<vector<Point2f>> imagePoints; // ��Ŷ�ά����
//    // ������ά������������������ϵʱ��Ĺ�ϵ�����ڽǵ���������
//    for (int i = 0; i < imgs.size(); i++)
//    {
//        Mat rvec = rvecs[i], tvec = tvecs[i];
//        vector<Point3f> PointSets = objectPoints[i];
//        vector<Point2f> imagePoint;                                                 // ��Ŷ�ά����
//        projectPoints(PointSets, rvec, tvec, cameraMatrix, distCoeffs, imagePoint); // ������ά��/��ת��ƽ������/ǰ�ļ���õ����ڲξ���ͻ������/�����ά��
//        imagePoints.push_back(imagePoint);
//    }
//
//    // -8 ������ͶӰ���
//    vector<vector<double>> ReProjectionError;
//    vector<vector<double>> ReProjectionErrorX;
//    vector<vector<double>> ReProjectionErrorY;
//    double e = 0.0;
//    for (int i = 0; i < imgs.size(); i++)
//    {
//        vector<Point2f> imagePoint = imagePoints[i]; // ��Ŷ�ά����
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
//    // �������ֵ��ͼ���м������ʵʱ֮���ƽ�����
//    cout << "ƽ����ͶӰ���:" << e / (imagePoints[0].size() * imgs.size()) << endl;
//
//    // -9 ������ͶӰ�������
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
//    // Բ�α궨��ǵ���ȡ
//    Mat img2 = imread("../CalibrateData/circle.png");
//    Mat gray2;
//    cvtColor(img2, gray2, COLOR_BGR2GRAY);
//    Size board_size2 = Size(7, 7);          //Բ�α궨��Բ����Ŀ���У��У�
//    vector<Point2f> img2_points;//���ǵ㣬����ͼ�����vector<Point2f>
//    findCirclesGrid(gray2, board_size2, img2_points);           //����Բ�α궨����
//    find4QuadCornerSubpix(gray2, img2_points, Size(5, 5));      //ϸ��Բ�α궨��ǵ�����
//    drawChessboardCorners(img2, board_size2, img2_points, true);
//    imshow("Բ�α궨��ǵ�����", img2);
//    waitKey();
//    */
//
//    return 0;
//}

