#include "MTCNN.h"
#include "opencv2/dnn.hpp"

namespace mtcnn {
	int minSize = 200;
	double factor = 0.709;
	double threshold[3] = { 0.6, 0.7, 0.5 };

	cv::dnn::Net PNet_;
	cv::dnn::Net RNet_;
	cv::dnn::Net ONet_;

	std::vector<FaceInfo> condidate_rects_;
	std::vector<FaceInfo> total_boxes_;
	std::vector<FaceInfo> regressed_rects_;
	std::vector<FaceInfo> regressed_pading_;

	std::vector<cv::Mat> crop_img_;
	int curr_feature_map_w_;
	int curr_feature_map_h_;
	int num_channels_;
	// compare score
	bool CompareBBox(const FaceInfo & a, const FaceInfo & b) {
		return a.bbox.score > b.bbox.score;
	}

	MTCNN::MTCNN(const std::string & model_dir) {
		PNet_ = cv::dnn::readNetFromCaffe(model_dir + "/det1.prototxt", model_dir + "/det1.caffemodel");
		RNet_ = cv::dnn::readNetFromCaffe(model_dir + "/det2.prototxt", model_dir + "/det2.caffemodel");
		ONet_ = cv::dnn::readNetFromCaffe(model_dir + "/det3.prototxt", model_dir + "/det3.caffemodel");
	}

	void MTCNN::setMinSize(int size) {
		minSize = size;
	}

	void MTCNN::drawDection(cv::Mat& frame, std::vector<FaceInfo>& faceInfo)
	{
		for (int i = 0; i < faceInfo.size(); i++) {
			int x = faceInfo[i].bbox.y1;
			int y = faceInfo[i].bbox.x1;
			int h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
			int w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;
			cv::rectangle(frame, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
		}
		for (int i = 0; i < faceInfo.size(); i++) {
			FacePts facePts = faceInfo[i].facePts;
			for (int j = 0; j < 5; j++) {
				int x = facePts.y[j];
				int y = facePts.x[j];
				cv::circle(frame, cv::Point(x, y), 1, cv::Scalar(255, 255, 0), 2);
			}
		}
	}

	std::vector<FaceInfo> NonMaximumSuppression(std::vector<FaceInfo>& bboxes,
		float thresh, char methodType) {
		std::vector<FaceInfo> bboxes_nms;
		std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

		int32_t select_idx = 0;
		int32_t num_bbox = static_cast<int32_t>(bboxes.size());
		std::vector<int32_t> mask_merged(num_bbox, 0);
		bool all_merged = false;

		while (!all_merged) {
			while (select_idx < num_bbox && mask_merged[select_idx] == 1)
				select_idx++;
			if (select_idx == num_bbox) {
				all_merged = true;
				continue;
			}

			bboxes_nms.push_back(bboxes[select_idx]);
			mask_merged[select_idx] = 1;

			FaceRect select_bbox = bboxes[select_idx].bbox;
			float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
			float x1 = static_cast<float>(select_bbox.x1);
			float y1 = static_cast<float>(select_bbox.y1);
			float x2 = static_cast<float>(select_bbox.x2);
			float y2 = static_cast<float>(select_bbox.y2);

			select_idx++;
			for (int32_t i = select_idx; i < num_bbox; i++) {
				if (mask_merged[i] == 1)
					continue;

				FaceRect& bbox_i = bboxes[i].bbox;
				float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
				float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
				float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
				float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
				if (w <= 0 || h <= 0)
					continue;

				float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
				float area_intersect = w * h;

				switch (methodType) {
				case 'u':
					if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
						mask_merged[i] = 1;
					break;
				case 'm':
					if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
						mask_merged[i] = 1;
					break;
				default:
					break;
				}
			}
		}
		return bboxes_nms;
	}

	void Bbox2Square(std::vector<FaceInfo>& bboxes) {
		for (int i = 0; i < bboxes.size(); i++) {
			float h = bboxes[i].bbox.x2 - bboxes[i].bbox.x1;
			float w = bboxes[i].bbox.y2 - bboxes[i].bbox.y1;
			float side = h > w ? h : w;
			bboxes[i].bbox.x1 += (h - side)*0.5;
			bboxes[i].bbox.y1 += (w - side)*0.5;

			bboxes[i].bbox.x2 = (int)(bboxes[i].bbox.x1 + side);
			bboxes[i].bbox.y2 = (int)(bboxes[i].bbox.y1 + side);
			bboxes[i].bbox.x1 = (int)(bboxes[i].bbox.x1);
			bboxes[i].bbox.y1 = (int)(bboxes[i].bbox.y1);

		}
	}

	std::vector<FaceInfo> BoxRegress(std::vector<FaceInfo>& faceInfo, int stage) {
		std::vector<FaceInfo> bboxes;
		for (int bboxId = 0; bboxId < faceInfo.size(); bboxId++) {
			FaceRect faceRect;
			FaceInfo tempFaceInfo;
			float regw = faceInfo[bboxId].bbox.y2 - faceInfo[bboxId].bbox.y1;
			regw += (stage == 1) ? 0 : 1;
			float regh = faceInfo[bboxId].bbox.x2 - faceInfo[bboxId].bbox.x1;
			regh += (stage == 1) ? 0 : 1;
			faceRect.y1 = faceInfo[bboxId].bbox.y1 + regw * faceInfo[bboxId].regression[0];
			faceRect.x1 = faceInfo[bboxId].bbox.x1 + regh * faceInfo[bboxId].regression[1];
			faceRect.y2 = faceInfo[bboxId].bbox.y2 + regw * faceInfo[bboxId].regression[2];
			faceRect.x2 = faceInfo[bboxId].bbox.x2 + regh * faceInfo[bboxId].regression[3];
			faceRect.score = faceInfo[bboxId].bbox.score;

			tempFaceInfo.bbox = faceRect;
			tempFaceInfo.regression = faceInfo[bboxId].regression;
			if (stage == 3)
				tempFaceInfo.facePts = faceInfo[bboxId].facePts;
			bboxes.push_back(tempFaceInfo);
		}
		return bboxes;
	}

	// compute the padding coordinates (pad the bounding boxes to square)
	void Padding(int img_w, int img_h) {
		for (int i = 0; i < regressed_rects_.size(); i++) {
			FaceInfo tempFaceInfo;
			tempFaceInfo = regressed_rects_[i];
			tempFaceInfo.bbox.y2 = (regressed_rects_[i].bbox.y2 >= img_w) ? img_w : regressed_rects_[i].bbox.y2;
			tempFaceInfo.bbox.x2 = (regressed_rects_[i].bbox.x2 >= img_h) ? img_h : regressed_rects_[i].bbox.x2;
			tempFaceInfo.bbox.y1 = (regressed_rects_[i].bbox.y1 < 1) ? 1 : regressed_rects_[i].bbox.y1;
			tempFaceInfo.bbox.x1 = (regressed_rects_[i].bbox.x1 < 1) ? 1 : regressed_rects_[i].bbox.x1;
			regressed_pading_.push_back(tempFaceInfo);
		}
	}

	void GenerateBoundingBox(const cv::Mat confidence, const cv::Mat reg_box,
		float scale, float thresh, int image_width, int image_height) {
		int stride = 2;
		int cellSize = 12;

		int curr_feature_map_w_ = std::ceil((image_width - cellSize)*1.0 / stride) + 1;
		int curr_feature_map_h_ = std::ceil((image_height - cellSize)*1.0 / stride) + 1;

		//std::cout << "Feature_map_size:"<< curr_feature_map_w_ <<" "<<curr_feature_map_h_<<std::endl;
		int regOffset = curr_feature_map_w_*curr_feature_map_h_;
		// the first count numbers are confidence of face
		int count = curr_feature_map_w_*curr_feature_map_h_;
		const float* confidence_data = (float*)(confidence.data);
		confidence_data += count;
		const float* reg_data = (float*)(reg_box.data);

		condidate_rects_.clear();
		for (int i = 0; i < count; i++) {
			if (*(confidence_data + i) >= thresh) {
				int y = i / curr_feature_map_w_;
				int x = i - curr_feature_map_w_ * y;

				float xTop = (int)((x*stride + 1) / scale);
				float yTop = (int)((y*stride + 1) / scale);
				float xBot = (int)((x*stride + cellSize - 1 + 1) / scale);
				float yBot = (int)((y*stride + cellSize - 1 + 1) / scale);
				FaceRect faceRect;
				faceRect.x1 = xTop;
				faceRect.y1 = yTop;
				faceRect.x2 = xBot;
				faceRect.y2 = yBot;
				faceRect.score = *(confidence_data + i);
				FaceInfo faceInfo;
				faceInfo.bbox = faceRect;
				faceInfo.regression = cv::Vec4f(reg_data[i + 0 * regOffset], reg_data[i + 1 * regOffset], reg_data[i + 2 * regOffset], reg_data[i + 3 * regOffset]);
				condidate_rects_.push_back(faceInfo);
			}
		}
	}

	void ClassifyFace_MulImage(const std::vector<FaceInfo>& regressed_rects, cv::Mat &sample_single,
		cv::dnn::Net &net, double thresh, char netName) {
		condidate_rects_.clear();
		int numBox = regressed_rects.size();
		int input_width = 24;
		int input_height = 24;
		if (netName == 'o') {
			input_width = 48;
			input_height = 48;
		}
		// load crop_img data to datum
		std::vector<cv::Mat> inputs;
		for (int i = 0; i < numBox; i++) {
			int pad_top = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
			int pad_left = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
			int pad_right = std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);
			int pad_bottom = std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);

			cv::Mat crop_img = sample_single(cv::Range(regressed_pading_[i].bbox.y1 - 1, regressed_pading_[i].bbox.y2),
				cv::Range(regressed_pading_[i].bbox.x1 - 1, regressed_pading_[i].bbox.x2));
			cv::copyMakeBorder(crop_img, crop_img, pad_left, pad_right, pad_top, pad_bottom, cv::BORDER_CONSTANT, cv::Scalar(0));
			cv::Mat roi;
			cv::resize(crop_img, roi, cv::Size(input_width, input_height));
			inputs.push_back(roi);
		}
		cv::Mat blob_input = cv::dnn::blobFromImages(inputs, 1 / 127.5, cv::Size(), { 127.5,127.5,127.5 }, true);
		regressed_pading_.clear();
		net.setInput(blob_input, "data");
		std::string outPutLayerName = (netName == 'r' ? "conv5-2" : "conv6-2");
		std::vector<cv::String> targets_node;
		if (netName == 'r') {
			targets_node.push_back(outPutLayerName);
			targets_node.push_back("prob1");
		}
		else {
			targets_node.push_back(outPutLayerName);
			targets_node.push_back("prob1");
			targets_node.push_back("conv6-3");
		}
		std::vector< cv::Mat > targets_blobs;
		net.forward(targets_blobs, targets_node);
		cv::Mat confidence = targets_blobs[1];
		cv::Mat reg = targets_blobs[0];
		cv::Mat reg_landmark;

		const float* confidence_data = (float*)(confidence.data);
		const float* reg_data = (float*)(reg.data);
		const float* points_data = nullptr;
		if (netName == 'o') {
			reg_landmark = targets_blobs[2];
			points_data = (float*)(reg_landmark.data);
		}
		for (int i = 0; i < numBox; i++) {
			if (*(confidence_data + i * 2 + 1) > thresh) {
				FaceRect faceRect;
				faceRect.x1 = regressed_rects[i].bbox.x1;
				faceRect.y1 = regressed_rects[i].bbox.y1;
				faceRect.x2 = regressed_rects[i].bbox.x2;
				faceRect.y2 = regressed_rects[i].bbox.y2;
				faceRect.score = *(confidence_data + i * 2 + 1);
				FaceInfo faceInfo;
				faceInfo.bbox = faceRect;
				faceInfo.regression = cv::Vec4f(reg_data[4 * i + 0], reg_data[4 * i + 1], reg_data[4 * i + 2], reg_data[4 * i + 3]);

				// x x x x x y y y y y
				if (netName == 'o') {
					FacePts face_pts;
					float w = faceRect.y2 - faceRect.y1 + 1;
					float h = faceRect.x2 - faceRect.x1 + 1;
					for (int j = 0; j < 5; j++) {
						face_pts.y[j] = faceRect.y1 + *(points_data + j + 10 * i) * h - 1;
						face_pts.x[j] = faceRect.x1 + *(points_data + j + 5 + 10 * i) * w - 1;
					}
					faceInfo.facePts = face_pts;
				}
				condidate_rects_.push_back(faceInfo);
			}
		}
	}

	int MTCNN::detect(const cv::Mat& image, std::vector<FaceInfo>& faceInfo) {
		// 2~3ms
		// invert to RGB color space and float type
		cv::Mat sample_single;
		sample_single = image.t();

		int height = image.rows;
		int width = image.cols;
		int minWH = std::min(height, width);
		int factor_count = 0;
		double m = 12. / minSize;
		minWH *= m;
		std::vector<double> scales;
		while (minWH >= 12) {
			scales.push_back(m * std::pow(factor, factor_count));
			minWH *= factor;
			++factor_count;
		}
		// 11ms main consum
		total_boxes_.clear();
		for (int i = 0; i < factor_count; i++)
		{
			double scale = scales[i];
			int ws = std::ceil(height*scale);
			int hs = std::ceil(width*scale);

			// wrap image and normalization using INTER_AREA method
			cv::Mat inputBlob = cv::dnn::blobFromImage(sample_single, 1 / 127.5, cv::Size(ws, hs), {127.5,127.5,127.5}, true);

			float* c = (float*)inputBlob.data;
			PNet_.setInput(inputBlob, "data");
			const std::vector< cv::String >  targets_node{ "conv4-2","prob1" };
			std::vector< cv::Mat > targets_blobs;
			PNet_.forward(targets_blobs, targets_node);
			cv::Mat prob = targets_blobs[1];
			cv::Mat reg = targets_blobs[0];
			GenerateBoundingBox(prob, reg, scale, threshold[0], ws, hs);
			std::vector<FaceInfo> bboxes_nms = NonMaximumSuppression(condidate_rects_, 0.5, 'u');
			total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
		}
		int numBox = total_boxes_.size();
		if (numBox != 0) {
			total_boxes_ = NonMaximumSuppression(total_boxes_, 0.7, 'u');
			regressed_rects_ = BoxRegress(total_boxes_, 1);
			total_boxes_.clear();
			Bbox2Square(regressed_rects_);
			Padding(width, height);

			/// Second stage
			ClassifyFace_MulImage(regressed_rects_, sample_single, RNet_, threshold[1], 'r');
			condidate_rects_ = NonMaximumSuppression(condidate_rects_, 0.7, 'u');
			regressed_rects_ = BoxRegress(condidate_rects_, 2);

			Bbox2Square(regressed_rects_);
			Padding(width, height);
			/// three stage
			numBox = regressed_rects_.size();
			if (numBox != 0) {
				ClassifyFace_MulImage(regressed_rects_, sample_single, ONet_, threshold[2], 'o');
				regressed_rects_ = BoxRegress(condidate_rects_, 3);
				faceInfo = NonMaximumSuppression(regressed_rects_, 0.7, 'm');
			}
		}
		regressed_pading_.clear();
		regressed_rects_.clear();
		condidate_rects_.clear();
		return 0;
	}
}