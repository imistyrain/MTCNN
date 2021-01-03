#include "mtcnn.h"
#include "mropencv.h"

#if _WIN32
	#pragma comment(lib,"ncnn.lib")
#endif

cv::Mat drawDetection(const cv::Mat &img, std::vector<Bbox> &box){
    cv::Mat show = img.clone();
    const int num_box = box.size();
    std::vector<cv::Rect> bbox;
    bbox.resize(num_box);
    for (int i = 0; i < num_box; i++) {
        bbox[i] = cv::Rect(box[i].x1, box[i].y1, box[i].x2 - box[i].x1 + 1, box[i].y2 - box[i].y1 + 1);
        for (int j = 0; j < 5; j = j + 1){
            cv::circle(show, cv::Point(box[i].ppoint[j], box[i].ppoint[j + 5]), 2, {0, 0, 255});
        }
    }
    for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
        rectangle(show, (*it), {0, 255, 0}, 2);
    }
    return show;
}

void test_camera(MTCNN &mtcnn){
    cv::VideoCapture mVideoCapture(0);
    cv::Mat frame;
    mVideoCapture >> frame;
    std::vector<Bbox> faces;
    while (!frame.empty()) {
        mVideoCapture >> frame;
        if (frame.empty()) {
            break;
        }
        cv::TickMeter tm;
        tm.start();
        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
        mtcnn.detect(ncnn_img, faces);
        tm.stop();
        cv::Mat show = drawDetection(frame, faces);
        cv::putText(show, to_string(tm.getTimeMilli()) + "ms", {0, 40}, 3, 1, { 0,0,255 });
        cv::imshow("img", show);
        cv::waitKey(1);
    }
    return;
}

int main(int argc, char** argv){
	MTCNN mtcnn;
	mtcnn.init("../model/ncnn");
	mtcnn.SetMinFace(160);
    test_camera(mtcnn);
    return 0;
}