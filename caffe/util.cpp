#include"util.h"
#include "mrutil.h"


std::vector<cv::Mat> Align5points(const cv::Mat &img, const std::vector<mtcnn::FaceInfo>&faceInfo){
	std::vector<cv::Point2f>  p2s;
	p2s.push_back(cv::Point2f(30.2946, 51.6963));
	p2s.push_back(cv::Point2f(65.5318, 51.5014));
	p2s.push_back(cv::Point2f(48.0252, 71.7366));
	p2s.push_back(cv::Point2f(33.5493, 92.3655));
	p2s.push_back(cv::Point2f(62.7299, 92.2041));
	std::vector<cv::Mat>dsts;
	for (int i = 0; i < faceInfo.size(); i++){
		std::vector<cv::Point2f> p1s;
		mtcnn::FacePts facePts = faceInfo[i].facePts;
		for (int j = 0; j < 5; j++){
			p1s.push_back(cv::Point(facePts.y[j], facePts.x[j]));
		}
		cv::Mat t = cv::estimateRigidTransform(p1s, p2s, false);
		if (!t.empty()){
			cv::Mat dst;
			cv::warpAffine(img, dst, t, cv::Size(96, 112));
			dsts.push_back(dst);
		} else {
			dsts.push_back(img);
		}
	}
	return dsts;
}

void drawArrow(cv::Mat image, cv::Point2i p, cv::Point2i q, cv::Scalar color,  int arrowMagnitude, int thickness, int line_type, int shift){
    //Draw the principle line
    cv::line(image, p, q, color, thickness, line_type, shift);
    const double PI = CV_PI;
    //compute the angle alpha
    double angle = atan2((double)p.y-q.y, (double)p.x-q.x);
    //compute the coordinates of the first segment
    p.x = (int) ( q.x +  arrowMagnitude * cos(angle + PI/4));
    p.y = (int) ( q.y +  arrowMagnitude * sin(angle + PI/4));
    //Draw the first segment
    cv::line(image, p, q, color, thickness, line_type, shift);
    //compute the coordinates of the second segment
    p.x = (int) ( q.x +  arrowMagnitude * cos(angle - PI/4));
    p.y = (int) ( q.y +  arrowMagnitude * sin(angle - PI/4));
    //Draw the second segment
    cv::line(image, p, q, color, thickness, line_type, shift);
}

void draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f> &list_points2d){
    cv::Scalar red(0, 0, 255);
    cv::Scalar green(0,255,0);
    cv::Scalar blue(255,0,0);
    cv::Scalar black(0,0,0);

    cv::Point2i origin = list_points2d[0];
    cv::Point2i pointX = list_points2d[1];
    cv::Point2i pointY = list_points2d[2];
    cv::Point2i pointZ = list_points2d[3];

    drawArrow(image, origin, pointX, red, 9, 2);
    drawArrow(image, origin, pointY, green, 9, 2);
    drawArrow(image, origin, pointZ, blue, 9, 2);
    cv::circle(image, origin, 2, black, -1 );
}

void drawEAV(cv::Mat &img,cv::Vec3f &eav){
	cv::putText(img,"x:"+double2string(eav[0]),cv::Point(0,30),3,1,cv::Scalar(0,0,255));
	cv::putText(img,"y:"+double2string(eav[1]),cv::Point(0,60),3,1,cv::Scalar(0,255,0));
	cv::putText(img,"z:"+double2string(eav[2]),cv::Point(0,90),3,1,cv::Scalar(255,0,0));
}