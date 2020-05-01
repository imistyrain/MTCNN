#include "MTCNN.h"
#include "mrdir.h"
#include "mropencv.h"
#include "mrutil.h"
#include "algorithm"
#if WIN32
#pragma comment( lib, cvLIB("calib3d"))
#endif
using namespace mtcnn;
using namespace std;
const std::string rootdir = "../";
const std::string imgdir = rootdir+"/imgs";
const std::string resultdir = rootdir + "/results";
const std::string proto_model_dir = rootdir + "/model/caffe";

#if _WIN32
const string casiadir = "E:/Face/Datasets/CASIA-maxpy-clean";
const string outdir = "E:/Face/Datasets/CASIA-mtcnn";
#else
const string casiadir = "~\CASIA-maxpy-clean";
const string outdir = "~\CASIA-mtcnn";
#endif

vector<cv::Mat> Align5points(const cv::Mat &img, const std::vector<FaceInfo>&faceInfo)
{
	std::vector<cv::Point2f>  p2s;
	p2s.push_back(cv::Point2f(30.2946, 51.6963));
	p2s.push_back(cv::Point2f(65.5318, 51.5014));
	p2s.push_back(cv::Point2f(48.0252, 71.7366));
	p2s.push_back(cv::Point2f(33.5493, 92.3655));
	p2s.push_back(cv::Point2f(62.7299, 92.2041));
	vector<Mat>dsts;
	for (int i = 0; i < faceInfo.size(); i++)
	{
		std::vector<cv::Point2f> p1s;
		FacePts facePts = faceInfo[i].facePts;
		for (int j = 0; j < 5; j++)
		{
			p1s.push_back(cv::Point(facePts.y[j], facePts.x[j]));
		}
		cv::Mat t = cv::estimateRigidTransform(p1s, p2s, false);
		if (!t.empty())
		{
			Mat dst;
			cv::warpAffine(img, dst, t, Size(96, 112));
			dsts.push_back(dst);
		}
		else
		{
			dsts.push_back(img);
		}
	}
	return dsts;
}

class PoseEstimator
{
public:
    cv::Vec3d estimateHeadPose(const cv::Mat &img, const vector<Point2f > &imagePoints);
    PoseEstimator() { init(); }
private:
    std::vector<cv::Point3f > modelPoints;
    void init();
};
void PoseEstimator::init()
{
    modelPoints.push_back(Point3f(2.37427, 110.322, 21.7776));	// l eye (v 314)
    modelPoints.push_back(Point3f(70.0602, 109.898, 20.8234));	// r eye (v 0)
    modelPoints.push_back(Point3f(36.8301, 78.3185, 52.0345));	//nose (v 1879)
    modelPoints.push_back(Point3f(14.8498, 51.0115, 30.2378));	// l mouth (v 1502)
    modelPoints.push_back(Point3f(58.1825, 51.0115, 29.6224));	// r mouth (v 695)   
}
cv::Vec3d PoseEstimator::estimateHeadPose(const cv::Mat &img,const vector<Point2f > &imagePoints)
{
    cv::Vec3d eav;
    cv::Mat op = cv::Mat(modelPoints);
    std::vector<double> rv(3), tv(3);
    cv::Mat rvec = cv::Mat(rv);
    cv::Mat tvec = Mat(tv);
    double _d[9] = { 1,0,0,0,-1,0,0,0,-1 };
    double rot[9] = { 0 };
    cv::Mat camMatrix = Mat(3, 3, CV_64FC1);
    cv::Mat ip(imagePoints);
    int max_d = (std::max)(img.rows, img.cols);
    camMatrix = (Mat_<double>(3, 3) << max_d, 0, img.cols / 2.0,0, max_d, img.rows / 2.0,0, 0, 1.0);
    double _dc[] = { 0,0,0,0 };
    solvePnP(op, ip, camMatrix, Mat(1, 4, CV_64FC1, _dc), rvec, tvec, false, CV_EPNP);
#if 1
    cv::Mat rotM(3, 3, CV_64FC1, rot);
    cv::Rodrigues(rvec, rotM);
    double* _r = rotM.ptr<double>();
    double _pm[12] = { _r[0],_r[1],_r[2],tv[0],
        _r[3],_r[4],_r[5],tv[1],
        _r[6],_r[7],_r[8],tv[2] };
    Matx34d P(_pm);
    Mat KP = camMatrix * Mat(P);
    for (int i = 0; i < op.rows; i++) {
        Mat_<double> X = (Mat_<double>(4, 1) << op.at<float>(i, 0), op.at<float>(i, 1), op.at<float>(i, 2), 1.0);
        Mat_<double> opt_p = KP * X;
        Point2f opt_p_img(opt_p(0) / opt_p(2), opt_p(1) / opt_p(2));
        circle(img, opt_p_img, 4, Scalar(0, 0, 255), 1);
    }
#endif
    for(int i=0;i<3;i++)
        eav[i] = rvec.at<float>(0, i);
    return eav;
}

int testcamera(int cameraindex = 0)
{
    PoseEstimator pe;
    MTCNN detector(proto_model_dir);
    cv::VideoCapture cap(cameraindex);
    cv::Mat frame;
    while (cap.read(frame)) {
        std::vector<FaceInfo> faceInfo;
        TickMeter tm;
        tm.start();
        detector.Detect(frame, faceInfo);
        tm.stop();
        //cout << tm.getTimeMilli() << "ms" << endl;
        for (int i = 0; i < faceInfo.size(); i++)
        {
            vector<Point2f > imagePoints;
            auto fi = faceInfo[0];
            for (int i = 0; i < 5; i++)
            {
                imagePoints.push_back(cv::Point2f(fi.facePts.y[i], fi.facePts.x[i]));
            }
            auto eav=pe.estimateHeadPose(frame, imagePoints);
            cout << eav << endl;
        }
        MTCNN::drawDectionResult(frame, faceInfo);
        cv::imshow("img", frame);
        if ((char)cv::waitKey(1) == 'q')
            break;
    }
    return 0;
}

int testdir()
{
	MTCNN detector(proto_model_dir);
    PoseEstimator pe;
	vector<string>files=getAllFilesinDir(imgdir);
	cv::Mat frame;

	for (int i = 0; i < files.size(); i++)
	{
		string imageName = imgdir + "/" + files[i];
        std::cout << imageName;
		frame=cv::imread(imageName);
        if(!frame.data)
            continue;
		clock_t t1 = clock();
		std::vector<FaceInfo> faceInfo;
		detector.Detect(frame, faceInfo);
		std::cout << " : " << (clock() - t1)*1.0 / 1000 << std::endl;
// 		vector<Mat> alignehdfaces = Align5points(frame,faceInfo);
// 		for (int j = 0; j < alignehdfaces.size(); j++)
// 		{
// 			string alignpath="align/"+int2string(j)+"_"+files[i];
// 			imwrite(alignpath, alignehdfaces[j]);
// 		}
//         vector<Point2f > imagePoints;
//         auto fi = faceInfo[0];
//         for (int i = 0; i < 5; i++)
//         {
//             imagePoints.push_back(cv::Point2f(fi.facePts.y[i], fi.facePts.x[i]));
//         }
//         pe.estimateHeadPose(frame, imagePoints);
		MTCNN::drawDectionResult(frame,faceInfo);
		cv::imshow("img", frame);
		string resultpath = resultdir + "/"+files[i];
		cv::imwrite(resultpath, frame);
		cv::waitKey(1);
	}
	return 0;
}

int testibm()
{
	MTCNN detector(proto_model_dir);
	vector<string>files=getAllFilesinDir(imgdir);
	cv::Mat frame;
	for (int i = 0; i < files.size(); i++)
	{
		string imageName = imgdir + "/" + files[i];
		frame = cv::imread(imageName);
		clock_t t1 = clock();
		std::vector<FaceInfo> faceInfo;
		detector.Detect(frame, faceInfo);
		std::cout << "Detect Using: " << (clock() - t1)*1.0 / 1000 << std::endl;
		MTCNN::drawDectionResult(frame, faceInfo);
		cv::imshow("img", frame);
		string resultpath = resultdir + "/" + files[i];
		cv::imwrite(resultpath, frame);
		cv::waitKey(1);
	}
	return 0;
}
#define SAVE_FDDB_RESULTS 1
int eval_fddb()
{
	const char* fddb_dir = "E:/Face/Datasets/fddb";
	string format = fddb_dir + string("/MTCNN/%Y%m%d-%H%M%S");
	time_t t = time(NULL);
	char buff[300];
	strftime(buff, sizeof(buff), format.c_str(), localtime(&t));
	if (SAVE_FDDB_RESULTS) {
		MKDIR(buff);
	}
	string result_prefix(buff);
	string prefix = fddb_dir + string("/images/");
	MTCNN detector(proto_model_dir);
	int counter = 0;
//#pragma omp parallel for
	for (int i = 1; i <= 10; i++) 
	{
		char fddb[300];
		char fddb_out[300];
		char fddb_answer[300];
		cout<<"Folds: "<<i<<endl;
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
			string full_path = prefix + string(path) + string(".jpg");
			Mat img = imread(full_path);
			if (!img.data) {
				cout << "Cannot read " << full_path << endl;;
				continue;
			}
			clock_t t1 = clock();
			std::vector<FaceInfo> faceInfo;
			detector.Detect(img, faceInfo);
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
				FacePts facePts = faceInfo[t].facePts;
				for (int j = 0; j < 5; j++)
					cv::circle(img, cv::Point(facePts.y[j], facePts.x[j]), 1, cv::Scalar(255, 255, 0), 2);
			}
			cv::imshow("img", img);
			cv::waitKey(1);
			char buff[300];
			if (SAVE_FDDB_RESULTS) {
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
					cv::ellipse(img, Point2d(center_x, center_y), Size(major_axis_radius, minor_axis_radius), \
						angle, 0., 360., Scalar(255, 0, 0), 2);					
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

int extractCASIA()
{
	::google::InitGoogleLogging("");
	MTCNN detector(proto_model_dir);
	vector<string>subdirs=getAllSubdirs(casiadir);
	MKDIR(outdir.c_str());
	for (int i =0; i < subdirs.size(); i++)
	{
		string subdir = casiadir + "/" + subdirs[i];
		vector<string>files=getAllFilesinDir(subdir);
		string outsubdir = outdir + "/" + subdirs[i];
		MKDIR(outsubdir.c_str());
		for (int j = 0; j < files.size(); j++)
		{
			cout << i <<":"<<subdirs[i]<< " " << j << endl;
			string filepath = subdir + "/" + files[j];
			std::vector<FaceInfo> faceInfo;
			Mat frame = imread(filepath);
			detector.Detect(frame, faceInfo);
			if (faceInfo.size()>0)
			{
				int maxindex = 0, maxarea = 0;
				for (int k = 0; k < faceInfo.size(); k++)
				{
					auto bbox = faceInfo[k].bbox;
					int area = (bbox.x2 - bbox.x1)*(bbox.x2 - bbox.x1) + (bbox.y2 - bbox.y1)*(bbox.y2 - bbox.y1);
					if (area>maxarea)
					{
						maxarea = area;
						maxindex = k;
					}
				}
				vector<FaceInfo>maxface;
				maxface.push_back(faceInfo[maxindex]);
				vector<Mat> alignedfaces = Align5points(frame, maxface);
				if (alignedfaces.size() > 0)
				{
					string outpath = outsubdir + "/" + files[j];
					imwrite(outpath, alignedfaces[0]);
				}
				else
				{
					cout << "No Aligend Face" << endl;
				}
			}
			else
			{
				cout << "No Faces detected" << endl;
			}
		}
	}
	return 0;
}
int main(int argc, char **argv)
{
    testcamera();
//	testdir();
//	testibm();
//	eval_fddb();
//	extractCASIA();
}