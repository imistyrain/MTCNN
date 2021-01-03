#pragma once
#include "vector"
#include "opencv2/opencv.hpp"
#include "MTCNN.h"

std::vector<cv::Mat> Align5points(const cv::Mat &img, const std::vector<mtcnn::FaceInfo>&faceInfo);

void drawArrow(cv::Mat image, cv::Point2i p, cv::Point2i q, cv::Scalar color, int arrowMagnitude = 9, int thickness = 1, int line_type = 8, int shift = 0);
void draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f>& list_points2d);
void drawEAV(cv::Mat &img,cv::Vec3f &eav);