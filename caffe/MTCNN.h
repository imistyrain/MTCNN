#pragma once
#include "opencv2/opencv.hpp"

namespace mtcnn
{
	typedef struct{
		float x1;
		float y1;
		float x2;
		float y2;
		float score;
	} FaceRect;

	typedef struct{
		float x[5], y[5];
	} FacePts;

	typedef struct{
		FaceRect bbox;
		cv::Vec4f regression;
		FacePts facePts;
		double roll;
		double pitch;
		double yaw;
	} FaceInfo;

	class MTCNN {
	public:
		MTCNN(const std::string& model_dir);
		void setMinSize(int size = 200);
		int detect(const cv::Mat& img, std::vector<FaceInfo> &faceInfo);
		static void drawDection(cv::Mat &img, std::vector<FaceInfo> &faceInfo);
	};
}