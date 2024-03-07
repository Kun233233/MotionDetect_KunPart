//
// Created by root on 9/17/17.
//

#include <stdio.h>

#include "depth_to_color.h"
#include "flash_bin_parser.h"
#include "calibration_params.h"
#include "calibration_params_parser.h"
#include <opencv2/opencv.hpp>

#include "Eigen/dense"


#define LOAD_INI_FILE
//#define LOAD_JSON_FILE

int main(int argc, char *argv [])
{
    DepthToColor d2c;
    printf("Version: %s\n", d2c.GetVersion());

    if (6 > argc)
    {
        fprintf(stderr, "Usage: %s color_img depth_img camera_param isMirror depth_unit [ir_img]\n", argv[0]);
        return -1;
    }

    const char *color_img_fname = argv[1];
    const char *depth_img_fname = argv[2];
    char *camera_param = argv[3];
	float depth_unit = 1.0;
	sscanf(argv[5], "%f", &depth_unit);
	int isMirror = 0;
	sscanf(argv[4],"%d",&isMirror);
	const char *ir_img_fname =nullptr;
	if(argc == 7 )
		ir_img_fname = argv[6];

#ifdef LOAD_INI_FILE
	bool success = d2c.LoadParameters(camera_param);
#else	// load SW_D2C from binary file/buffer


	SW_D2C soft_d2c = {0};

#ifdef LOAD_JSON_FILE
	CalibrateParams calib_params;
	ParamParseErr err = LoadCalibrationParams(camera_param, &calib_params, CALIB_PARAM_FORMAT_JSON);
	if (err != ParamParseErr::PARAM_PARSE_OK)
	{
		printf("err: %d\n", err);
		return false;
	}
	//float d_intr_p[4];//[fx,fy,cx,cy]
	//float c_intr_p[4];//[fx,fy,cx,cy]
	//float d2c_r[9];//[r00,r01,r02;r10,r11,r12;r20,r21,r22]
	//float d2c_t[3];//[t1,t2,t3]
	//float d_k[5];//[k1,k2,p1,p2,k3]
	//float c_k[5];
	double scale = 1.0;
	soft_d2c.d_intr_p[0] = calib_params.stereo_params_.ir_intrin_.focal_x_ * scale;
	soft_d2c.d_intr_p[1] = calib_params.stereo_params_.ir_intrin_.focal_y_ * scale;
	soft_d2c.d_intr_p[2] = calib_params.stereo_params_.ir_intrin_.cx_ * scale;
	soft_d2c.d_intr_p[3] = calib_params.stereo_params_.ir_intrin_.cy_ * scale;
	soft_d2c.c_intr_p[0] = calib_params.stereo_params_.rgb_intrin_.focal_x_ * scale;
	soft_d2c.c_intr_p[1] = calib_params.stereo_params_.rgb_intrin_.focal_y_ * scale;
	soft_d2c.c_intr_p[2] = calib_params.stereo_params_.rgb_intrin_.cx_ * scale;
	soft_d2c.c_intr_p[3] = calib_params.stereo_params_.rgb_intrin_.cy_ * scale;
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotMat = Eigen::Matrix3d::Identity();
	rotMat = Eigen::AngleAxisd(calib_params.stereo_params_.rgb_2_ir_pose_.rz_, Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(calib_params.stereo_params_.rgb_2_ir_pose_.rx_, Eigen::Vector3d::UnitX()) *
		Eigen::AngleAxisd(calib_params.stereo_params_.rgb_2_ir_pose_.ry_, Eigen::Vector3d::UnitY());
	int idx = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			soft_d2c.d2c_r[3 * i + j] = rotMat(i, j);
		}
	}
	soft_d2c.d2c_t[0] = calib_params.stereo_params_.rgb_2_ir_pose_.tx_;
	soft_d2c.d2c_t[1] = calib_params.stereo_params_.rgb_2_ir_pose_.ty_;
	soft_d2c.d2c_t[2] = calib_params.stereo_params_.rgb_2_ir_pose_.tz_;

	soft_d2c.d_k[0] = calib_params.stereo_params_.ir_intrin_.k1_;
	soft_d2c.d_k[1] = calib_params.stereo_params_.ir_intrin_.k2_;
	soft_d2c.d_k[2] = calib_params.stereo_params_.ir_intrin_.p1_;
	soft_d2c.d_k[3] = calib_params.stereo_params_.ir_intrin_.p2_;
	soft_d2c.d_k[4] = calib_params.stereo_params_.ir_intrin_.k3_;
	soft_d2c.c_k[0] = calib_params.stereo_params_.rgb_intrin_.k1_;
	soft_d2c.c_k[1] = calib_params.stereo_params_.rgb_intrin_.k2_;
	soft_d2c.c_k[2] = calib_params.stereo_params_.rgb_intrin_.p1_;
	soft_d2c.c_k[3] = calib_params.stereo_params_.rgb_intrin_.p2_;
	soft_d2c.c_k[4] = calib_params.stereo_params_.rgb_intrin_.k3_;

	// fx fy cx cy
	printf("\n soft_d2c depth K\n");
	for (int i = 0; i < 4; i++)
	{
		printf("soft_d2c.d_intr_p[%d] = %f\n",i, soft_d2c.d_intr_p[i]);
		//printf("soft_d2c.c_intr_p[%d] = %f\n", i, soft_d2c.c_intr_p[i]);
	}
	printf("\nsoft_d2c color K\n");
	for (int i = 0; i < 4; i++)
	{
		//printf("soft_d2c.d_intr_p[%d] = %f\n", i, soft_d2c.d_intr_p[i]);
		printf("soft_d2c.c_intr_p[%d] = %f\n", i, soft_d2c.c_intr_p[i]);
	}

	//rot matrix
	printf("\nsoft_d2c rotMatrix\n");
	for (int i = 0; i < 9; i++)
	{
		printf("soft_d2c.d2c_r[%d] = %lf\n",i, soft_d2c.d2c_r[i]);
	}
	// translate
	printf("\nsoft_d2c translate\n");
	for (int i = 0; i < 3; i++)
	{
		printf("soft_d2c.d2c_t[%d] = %f\n",i, soft_d2c.d2c_t[i]);
	}

	//distortion
	printf("\nsoft_d2c distortion [k1,k2,p1,p2,k3] \n");

	for (int i = 0; i < 5; i++)
	{
		printf("soft_d2c.d_k[%d] = %f\n",i, soft_d2c.d_k[i]);
	}
	for (int i = 0; i < 5; i++)
	{
		printf("soft_d2c.c_k[%d] = %f\n", i, soft_d2c.c_k[i]);
	}
	

#else //load D2C params from flash_bin file
	Flash_bin_parser flashParser;

	FILE* fp = fopen(camera_param, "rb");

	printf("buffer size = %d \n", flashParser.getDataSize());

	char* buffer = new char[flashParser.getDataSize()];

	fread(buffer, 1, flashParser.getDataSize(), fp);


	fclose(fp);

	flashParser.parse_buffer_to_params(buffer, &soft_d2c);
	//flashParser.parse_bin_to_params(camera_param, &soft_d2c);
	printf("soft_d2c:d_intr_p[]\n");
	printf("%f %f %f %f \n", soft_d2c.d_intr_p[0], soft_d2c.d_intr_p[1], soft_d2c.d_intr_p[2], soft_d2c.d_intr_p[3]);
	printf("soft_d2c:c_intr_p[]\n");
	printf("%f %f %f %f \n", soft_d2c.c_intr_p[0], soft_d2c.c_intr_p[1], soft_d2c.c_intr_p[2], soft_d2c.c_intr_p[3]);
	delete[] buffer;

#endif 
	bool success = d2c.LoadParameters(&soft_d2c);

#endif
    if (!success)
    {
        fprintf(stderr, "LoadParameters failed!\n");
        return -1;
    }

    cv::Mat color_img = cv::imread(color_img_fname, cv::IMREAD_COLOR);
    //cv::cvtColor(color_img, color_img, CV_BGR2RGB);
    cv::Mat depth_img = cv::imread(depth_img_fname, cv::IMREAD_UNCHANGED);
	printf("ir_img_fname: %s\n", ir_img_fname);
	cv::Mat ir_img;
	if (ir_img_fname != nullptr)
	{
		ir_img = cv::imread(ir_img_fname, cv::IMREAD_UNCHANGED); 
		cv::imshow("ir", ir_img);
		cv::waitKey(30);
	}

	//cv::imshow("depth",depth_img);
	//cv::waitKey(0);
	

    int depth_width = depth_img.cols;
    int depth_height = depth_img.rows;
    int color_width = color_img.cols;
    int color_height = color_img.rows;

    int size = color_width * color_height*3 * sizeof(uint8_t); // 最大为rgb888模式的图像
    uint16_t * aligned_depth = new uint16_t[size];


	// 每个深度单位=1.0 mm
	success = d2c.SetDepthUnit(depth_unit);
	if (!success)
	{
		fprintf(stderr, "SetDepthUnit failed!\n");
		return -1;
	}

	#开软件D2C用
    success = d2c.PrepareDepthResolution(depth_width, depth_height);
    if (!success)
    {
        fprintf(stderr, "PrepareDepthResolution failed!\n");
        return -1;
    }

	#开软件ir2color用（输入为RGB分辨率）
	success = d2c.PrepareIrResolution(color_width, color_height);
	if (!success)
	{
		fprintf(stderr, "PrepareIrResolution failed!\n");
		return -1;
	}

    // 开畸变
    success = d2c.EnableDistortion(false);
    if (!success)
    {
        fprintf(stderr, "EnableDistortion failed!\n");
        return -1;
    }

	// fill gaps
	success = d2c.EnableDepthGapFill(true);
	if (!success)
	{
		fprintf(stderr, "EnableDepthGapFill failed!\n");
		return -1;
	}

	// 设置输入图像的镜像模式(深度和IR一致)
	success = d2c.SetMirrorMode(isMirror);
	if (!success)
	{
		fprintf(stderr, "SetMirrorMode failed!\n");
		return -1;
	}

    // 设置原始深度图中，感兴趣的深度范围
    success = d2c.SetDepthRange(100.0, 9000.0);
    if (!success)
    {
        fprintf(stderr, "SetDepthRange failed!\n");
        return -1;
    }
	//if (isMirror)
	//{
	//	cv::flip(depth_img, depth_img, 1); // flip Horizontal
	//}
    int ret = d2c.D2C(depth_img.ptr<uint16_t>(), depth_width, depth_height,
                      aligned_depth, color_width, color_height);
	
	cv::Mat aligned_img(cv::Size(color_width,color_height), CV_16UC1, aligned_depth);
	//if (isMirror)
	//{
	//	cv::flip(aligned_img, aligned_img, 1); // flip Horizontal
	//}

	std::vector<int> png_params;
	png_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	png_params.push_back(0);    // 无损

	std::string depthfile = depth_img_fname;
	int pos = depthfile.find_last_of('.');
	std::string depth_aligned_file = depthfile.substr(0, pos);
	depth_aligned_file += "_aligned.png";

	cv::imwrite(depth_aligned_file.c_str(), aligned_img, png_params);
    if (ret)
    {
        printf("D2C failed!\n");
        delete [] aligned_depth;
        aligned_depth = nullptr;
        return -1;
	}
	if (ir_img_fname != nullptr)
	{
		int depth_value = 597;
		for (int it = 0; it < 10; it++)
		{
			// disable distortion
			//success = d2c.EnableDistortion(false);
			if (!success)
			{
				fprintf(stderr, "EnableDepthGapFill failed!\n");
				return -1;
			}
			enum IR_TYPE irType = IR_8U;
			if (ir_img.depth() == CV_8U && ir_img.channels() == 1) // 8Bit-gray
			{
				printf("IR depth: CV_8UC1\n");
				irType = IR_8U;
				int ret = d2c.D2C_IR(ir_img.ptr<uint8_t>(), depth_value, depth_width, depth_height,
					(uint8_t*)aligned_depth, color_width, color_height, irType);
				if (ret)
				{
					printf("D2C_IR failed %d \n", ret);
				}
				else
				{
					cv::Mat ir_toColor(cv::Size(color_width, color_height), CV_8UC1, (uint8_t*)aligned_depth);
					//cv::Mat ir_toColor(cv::Size(color_width, color_height), CV_8UC1);
					cv::imshow("ir2c", ir_toColor);
					std::string ir_aligned_file = depthfile.substr(0, pos);
					ir_aligned_file += "_aligned_ir.png";
					cv::imwrite(ir_aligned_file, ir_toColor, png_params);
				}
			}
			else if (ir_img.depth() == CV_16U && ir_img.channels() == 1) // 16Bit-gray
			{
				printf("IR depth: CV_16UC1\n");
				irType = IR_16U;
				int ret = d2c.D2C_IR((uint8_t*)ir_img.ptr<uint16_t>(), depth_value, depth_width, depth_height,
					(uint8_t*)aligned_depth, color_width, color_height, irType);
				if (ret)
				{
					printf("D2C_IR failed %d \n", ret);
				}
				else
				{
					cv::Mat ir_toColor(cv::Size(color_width, color_height), CV_16UC1, (uint16_t*)aligned_depth);
					cv::imshow("ir2c", ir_toColor);
					std::string ir_aligned_file = depthfile.substr(0, pos);
					ir_aligned_file += "_aligned_ir.png";
					cv::imwrite(ir_aligned_file, ir_toColor, png_params);
				}
			}
			else if (ir_img.depth() == CV_8U && ir_img.channels() == 3) // RGB888
			{
				printf("IR depth: CV_8UC3\n");
				irType = IR_888;
				int ret = d2c.D2C_IR(ir_img.ptr<uint8_t>(), depth_value, depth_width, depth_height,
					(uint8_t*)aligned_depth, color_width, color_height, irType);
				if (ret)
				{
					printf("D2C_IR failed %d \n", ret);
				}
				else
				{
					cv::Mat ir_toColor(cv::Size(color_width, color_height), CV_8UC3, (uint8_t*)aligned_depth);
					cv::imshow("ir2c", ir_toColor);
					std::string ir_aligned_file = depthfile.substr(0, pos);
					ir_aligned_file += "_aligned_ir.png";
					cv::imwrite(ir_aligned_file, ir_toColor, png_params);
				}

			}
			depth_value += 60;
		}

	}

    delete [] aligned_depth;
    aligned_depth = nullptr;

    printf("D2C success!\n");

	cv::waitKey(0);
    return 0;
}
