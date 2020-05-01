#pragma once
#include <caffe/caffe.hpp>
#include "opencv2/opencv.hpp"

namespace mtcnn
{
	typedef struct FaceRect {
		float x1;
		float y1;
		float x2;
		float y2;
		float score; /**< Larger score should mean higher confidence. */
	} FaceRect;

	typedef struct FacePts {
		float x[5], y[5];
	} FacePts;

	typedef struct FaceInfo {
		FaceRect bbox;
		cv::Vec4f regression;
		FacePts facePts;
		double roll;
		double pitch;
		double yaw;
	} FaceInfo;
	class MTCNN {
	public:
		MTCNN(const std::string& proto_model_dir);
		int Detect(const cv::Mat& img, std::vector<FaceInfo> &faceInfo);
		void SetMinSize(const unsigned int size);
		int GetMinSize();
		static void drawDectionResult(cv::Mat &frame, std::vector<FaceInfo> &faceInfo);
	private:
		bool CvMatToDatumSignalChannel(const cv::Mat& cv_mat, caffe::Datum* datum);
		void WrapInputLayer(std::vector<cv::Mat>* input_channels, caffe::Blob<float>* input_layer,
			const int height, const int width);
		void GenerateBoundingBox(caffe::Blob<float>* confidence, caffe::Blob<float>* reg,
			float scale, float thresh, int image_width, int image_height);
		void ClassifyFace_MulImage(const std::vector<FaceInfo> &regressed_rects, cv::Mat &sample_single,
			boost::shared_ptr<caffe::Net<float> >& net, double thresh, char netName);
		std::vector<FaceInfo> NonMaximumSuppression(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
		void Bbox2Square(std::vector<FaceInfo>& bboxes);
		void Padding(int img_w, int img_h);
		std::vector<FaceInfo> BoxRegress(std::vector<FaceInfo> &faceInfo_, int stage);

	private:
		boost::shared_ptr<caffe::Net<float> > PNet_;
		boost::shared_ptr<caffe::Net<float> > RNet_;
		boost::shared_ptr<caffe::Net<float> > ONet_;

		// x1,y1,x2,t2 and score
		std::vector<FaceInfo> condidate_rects_;
		std::vector<FaceInfo> total_boxes_;
		std::vector<FaceInfo> regressed_rects_;
		std::vector<FaceInfo> regressed_pading_;

		std::vector<cv::Mat> crop_img_;
		int curr_feature_map_w_;
		int curr_feature_map_h_;
		int num_channels_;
		double threshold[3] = { 0.6, 0.7, 0.5 };
		double factor = 0.709;
		int minSize = 50;
	};
}