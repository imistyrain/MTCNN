#include "mtcnn.h"
#include <time.h>
#include "mropencv.h"

int mtcnnDetect(cv::Mat &image){
	if (!image.data)
		return -1;
	static mtcnn find;
    find.SetMinFaceSize(60);
	TickMeter tm;
	tm.start();
    std::vector<FaceInfo>fds;
	find.Detect(image,fds);
	tm.stop();
	cout << tm.getTimeMilli() << "ms" << endl;
	return 0;
}

int testimage(const string imgpath = "test.jpg"){
	cv::Mat image = cv::imread(imgpath);
	mtcnnDetect(image);
	cv::imshow("img", image);
	cv::waitKey();
	cv::imwrite("result.jpg",image);
	return 0;
}

int testcamera(int index=0){
	cv::Mat image;
	cv::VideoCapture cap(index);
	cap.set(3,640);
	cap.set(4,480);
	if (!cap.isOpened()) {
		cout << "fail to open camera " << index << endl;
		return -1;
	}
	while (cap.isOpened()){
		cap >> image;
		if (!image.data)
			break;
		if (image.cols!=640)
			cv::resize(image,image,cv::Size(640,480));
		mtcnnDetect(image);
		cv::imshow("mtcnn", image);
		cv::waitKey(1);
	}
    cv::waitKey();
    image.release();
    return 0;
}

int main(int argc, char*argv[]){
	if (argc > 1) {
		testimage(argv[1]);
	}
	if (0 != testcamera()) {
		testimage();
	}
	return 0;
}
