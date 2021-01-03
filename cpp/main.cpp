#include "time.h"
#include "MTCNN.h"
#include "mrdir.h"
#include "mrutil.h"
#include "mropencv.h"

const std::string rootdir = "../";
const std::string imgdir = rootdir+"/images";
const std::string resultdir = rootdir + "/results";
const std::string model_dir = rootdir + "/model/caffe";

std::vector<cv::Mat> Align5points(const cv::Mat &img, const std::vector<mtcnn::FaceInfo>&faceInfo)
{
	std::vector<cv::Point2f>  p2s;
	p2s.push_back(cv::Point2f(30.2946, 51.6963));
	p2s.push_back(cv::Point2f(65.5318, 51.5014));
	p2s.push_back(cv::Point2f(48.0252, 71.7366));
	p2s.push_back(cv::Point2f(33.5493, 92.3655));
	p2s.push_back(cv::Point2f(62.7299, 92.2041));
	std::vector<cv::Mat>dsts;
	for (int i = 0; i < faceInfo.size(); i++)
	{
		std::vector<cv::Point2f> p1s;
		mtcnn::FacePts facePts = faceInfo[i].facePts;
		for (int j = 0; j < 5; j++)
		{
			p1s.push_back(cv::Point(facePts.y[j], facePts.x[j]));
		}
		cv::Mat t = cv::estimateRigidTransform(p1s, p2s, false);
		if (!t.empty())
		{
			Mat dst;
			cv::warpAffine(img, dst, t, cv::Size(96, 112));
			dsts.push_back(dst);
		}
		else
		{
			dsts.push_back(img);
		}
	}
	return dsts;
}

void drawArrow(cv::Mat image, cv::Point2i p, cv::Point2i q, cv::Scalar color,  int arrowMagnitude = 9, int thickness=1, int line_type=8, int shift=0)
{
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

void draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f> &list_points2d)
{
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

class PoseEstimator{
public:
    cv::Vec3f estimateHeadPose(cv::Mat &img, const std::vector<Point2f > &imagePoints);
    PoseEstimator() { init(); }
private:
    std::vector<cv::Point3f > modelPoints;
    void init();
};

void PoseEstimator::init(){
    modelPoints.push_back(Point3f(2.37427, 110.322, 21.7776));	// l eye (v 314)
    modelPoints.push_back(Point3f(70.0602, 109.898, 20.8234));	// r eye (v 0)
    modelPoints.push_back(Point3f(36.8301, 78.3185, 52.0345));	//nose (v 1879)
    modelPoints.push_back(Point3f(14.8498, 51.0115, 30.2378));	// l mouth (v 1502)
    modelPoints.push_back(Point3f(58.1825, 51.0115, 29.6224));	// r mouth (v 695)   
}

cv::Vec3f PoseEstimator::estimateHeadPose(cv::Mat &img,const std::vector<Point2f > &imagePoints)
{
    cv::Mat rvec, tvec;
    int max_d = (img.rows + img.cols)/2;
    cv::Mat camMatrix = (Mat_<double>(3, 3) << max_d, 0, img.cols / 2.0,0, max_d, img.rows / 2.0, 0, 0, 1.0);
    solvePnP(modelPoints,imagePoints, camMatrix, cv::Mat(), rvec, tvec, false, CV_EPNP);
	cv::Mat R;
	cv::Rodrigues(rvec, R);
	std::vector<cv::Point3f> axises;
	std::vector<cv::Point2f> pts2d;
	float l = 40;
	int x = modelPoints[2].x;
	int y = modelPoints[2].y;
	int z = modelPoints[2].z;
	axises.push_back(cv::Point3f(x,y,z));
	axises.push_back(cv::Point3f(x+l,y,z));
	axises.push_back(cv::Point3f(x,y+l,z));
	axises.push_back(cv::Point3f(x,y,z+l));
	projectPoints(axises,rvec,tvec,camMatrix,cv::Mat(),pts2d);
	draw3DCoordinateAxes(img,pts2d);
	#if 0
		projectPoints(modelPoints,rvec,tvec,camMatrix,cv::Mat(),pts2d);
		for(int i = 0; i < pts2d.size(); i++){
			cv::circle(img,pts2d[i],5,cv::Scalar(255,0,0),-1);
		}
	#endif
	cv:Mat T;
	cv::Mat euler_angle;
	cv::Mat out_rotation, out_translation;
	cv::hconcat(R, tvec, T);
	cv::decomposeProjectionMatrix(T,camMatrix,out_rotation,out_translation,cv::noArray(),cv::noArray(),cv::noArray(),euler_angle);
	cv::Vec3f eav;
	for(int i = 0; i < 3; i++){
		eav[i] = euler_angle.at<double>(0,i);
	}
	drawEAV(img,eav);
	return eav;
}

int testcamera() {
	mtcnn::MTCNN detector(model_dir);
	PoseEstimator pe;
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cout << "Cannot open camera" << std::endl;
	}
    cv::Mat img;
	std::vector<mtcnn::FaceInfo> faceInfo;
    while (true) {
		cap >> img;
		if (!img.data) {
			break;
		}
        TickMeter tm;
        tm.start();
        detector.detect(img, faceInfo);
        tm.stop();
		std::string cost = double2string(tm.getTimeMilli()) + "ms";
        for (int i = 0; i < faceInfo.size(); i++){
			std::vector<cv::Point2f > imagePoints;
            auto fi = faceInfo[0];
            for (int i = 0; i < 5; i++){
                imagePoints.push_back(cv::Point2f(fi.facePts.y[i], fi.facePts.x[i]));
            }
            auto eav = pe.estimateHeadPose(img, imagePoints);
        }
		cv::putText(img, cost, {200, 40}, 3, 1, {0, 0, 255});
        mtcnn::MTCNN::drawDection(img, faceInfo);
        cv::imshow("img", img);
        cv::waitKey(1);
    }
    return 0;
}

int testdir(){
	mtcnn::MTCNN detector(model_dir);
    PoseEstimator pe;
	std::vector<std::string>files = getAllFilesinDir(imgdir);
	cv::Mat img;
	for (int i = 0; i < files.size(); i++){
		std::string imageName = imgdir + "/" + files[i];
        std::cout << files[i];
		img = cv::imread(imageName);
        if(!img.data)
            continue;
		clock_t t1 = clock();
		std::vector<mtcnn::FaceInfo> faceInfo;
		detector.detect(img, faceInfo);
		//std::cout << " : " << (clock() - t1)*1.0 / 1000 << std::endl;
		//std::vector<cv::Mat> alignehdfaces = Align5points(img, faceInfo);
		//for (int j = 0; j < alignehdfaces.size(); j++){
		//	std::string alignpath="align/"+int2string(j)+"_"+files[i];
		//	cv::imwrite(alignpath, alignehdfaces[j]);
		//}
		if (faceInfo.size() == 0) {
			std::cout << "No face detected" << std::endl;
			continue;
		}
        std::vector<cv::Point2f > imagePoints;
        auto fi = faceInfo[0];
        for (int i = 0; i < 5; i++){
            imagePoints.push_back(cv::Point2f(fi.facePts.y[i], fi.facePts.x[i]));
        }
        pe.estimateHeadPose(img, imagePoints);
		mtcnn::MTCNN::drawDection(img,faceInfo);
		std::string resultpath = resultdir + "/"+files[i];
		cv::imwrite(resultpath, img);
		cv::imshow("img", img);
		cv::waitKey(1);
	}
	cv::waitKey();
	return 0;
}

int eval_fddb(){
	const char* fddb_dir = "E:/Face/Datasets/fddb";
	std::string format = fddb_dir + std::string("/MTCNN/%Y%m%d-%H%M%S");
	time_t t = time(NULL);
	char buff[300];
	strftime(buff, sizeof(buff), format.c_str(), localtime(&t));
	MKDIR(buff);
	std::string result_prefix(buff);
	std::string prefix = std::string(fddb_dir) + "/images/";
	mtcnn::MTCNN detector(model_dir);
	int counter = 0;
//#pragma omp parallel for
	for (int i = 1; i <= 10; i++) 
	{
		char fddb[300];
		char fddb_out[300];
		char fddb_answer[300];
		std::cout<<"Folds: "<<i<< std::endl;
		sprintf(fddb, "%s/FDDB-folds/FDDB-fold-%02d.txt", fddb_dir, i);
		sprintf(fddb_out, "%s/MTCNN/fold-%02d-out.txt", fddb_dir, i);
		sprintf(fddb_answer, "%s/FDDB-folds/FDDB-fold-%02d-ellipseList.txt", fddb_dir, i);

		FILE* fin = fopen(fddb, "r");
		FILE* fanswer = fopen(fddb_answer, "r");
#ifdef _WIN32
		FILE* fout = fopen(fddb_out, "wb"); // replace \r\n on Windows platform		
#else
		FILE* fout = fopen(fddb_out, "w");	
#endif // WIN32
		
		char path[300];
		int counter = 0;
		while (fscanf(fin, "%s", path) > 0)
		{
			std::string full_path = prefix + std::string(path) + std::string(".jpg");
			cv::Mat img = imread(full_path);
			if (!img.data) {
				std::cout << "Cannot read " << full_path << std::endl;;
				continue;
			}
			clock_t t1 = clock();
			std::vector<mtcnn::FaceInfo> faceInfo;
			detector.detect(img, faceInfo);
			std::cout << "Detect " <<i<<": "<<counter<<" Using : " << (clock() - t1)*1.0 / 1000 << std::endl;
			const int n = faceInfo.size();
			fprintf(fout, "%s\n%d\n", path, n);
			for (int j = 0; j < n; j++) {
				int x = (int)faceInfo[j].bbox.x1;
				if (x < 0)x = 0;
				int y = (int)faceInfo[j].bbox.y1;
				if (y < 0)y = 0;
				int h = (int)faceInfo[j].bbox.x2 - faceInfo[j].bbox.x1 + 1;
				if (h>img.rows - x)h = img.rows - x;
				int w = (int)faceInfo[j].bbox.y2 - faceInfo[j].bbox.y1 + 1;
				if (w>img.cols-y)w = img.cols - y;
				float score = faceInfo[j].bbox.score;
				cv::rectangle(img, cv::Rect(y, x, w, h), cv::Scalar(0, 0, 255), 1);
				fprintf(fout, "%d %d %d %d %lf\n", y, x, w, h, score);
			}
			for (int t = 0; t < faceInfo.size(); t++){
				mtcnn::FacePts facePts = faceInfo[t].facePts;
				for (int j = 0; j < 5; j++)
					cv::circle(img, cv::Point(facePts.y[j], facePts.x[j]), 1, cv::Scalar(255, 255, 0), 2);
			}
			cv::imshow("img", img);
			cv::waitKey(1);
			char buff[300];
			if (1) {
				counter++;
				sprintf(buff, "%s/%02d_%03d.jpg", result_prefix.c_str(), i, counter);
				// get answer
				int face_n = 0;
				fscanf(fanswer, "%s", path);
				fscanf(fanswer, "%d", &face_n);
				for (int k = 0; k < face_n; k++)
				{
					double major_axis_radius, minor_axis_radius, angle, center_x, center_y, score;
					fscanf(fanswer, "%lf %lf %lf %lf %lf %lf", &major_axis_radius, &minor_axis_radius, \
						&angle, &center_x, &center_y, &score);
					// draw answer
					angle = angle / 3.1415926*180.;
					cv::ellipse(img, cv::Point2d(center_x, center_y), cv::Size(major_axis_radius, minor_axis_radius), \
						angle, 0., 360., { 255,0,0 }, 2);
				}
				cv::imwrite(buff, img);
			}
		}
		fclose(fin);
		fclose(fout);
		fclose(fanswer);
	}
	return 0;
}

int main(int argc, char **argv){
//	testcamera();
	testdir();
//	eval_fddb();
}