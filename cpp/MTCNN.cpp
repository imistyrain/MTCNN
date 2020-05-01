#include "MTCNN.h"
#include <caffe/layers/memory_data_layer.hpp>
#define SHOW_MTCNN_DEBUG 0
namespace mtcnn
{
	// compare score
	bool CompareBBox(const FaceInfo & a, const FaceInfo & b) {
		return a.bbox.score > b.bbox.score;
	}

	void MTCNN::SetMinSize(const unsigned int size)
	{
		minSize = size;
	}

	int MTCNN::GetMinSize()
	{
		return minSize;
	}

	void MTCNN::drawDectionResult(cv::Mat &frame, std::vector<FaceInfo> &faceInfo)
	{
		for (int i = 0; i < faceInfo.size(); i++){
			float x = faceInfo[i].bbox.x1;
			float y = faceInfo[i].bbox.y1;
			float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
			float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;
			cv::rectangle(frame, cv::Rect(y, x, w, h), cv::Scalar(255, 0, 0), 2);
		}
		for (int i = 0; i < faceInfo.size(); i++){
			FacePts facePts = faceInfo[i].facePts;
			for (int j = 0; j < 5; j++)
				cv::circle(frame, cv::Point(facePts.y[j], facePts.x[j]), 1, cv::Scalar(255, 255, 0), 2);
		}
	}

	// methodType : u is IoU(Intersection Over Union)
	// methodType : m is IoM(Intersection Over Maximum)
	std::vector<FaceInfo> MTCNN::NonMaximumSuppression(std::vector<FaceInfo>& bboxes,
		float thresh, char methodType){
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

	void MTCNN::Bbox2Square(std::vector<FaceInfo>& bboxes){
		for (int i = 0; i < bboxes.size(); i++){
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

	std::vector<FaceInfo> MTCNN::BoxRegress(std::vector<FaceInfo>& faceInfo, int stage){
		std::vector<FaceInfo> bboxes;
		for (int bboxId = 0; bboxId < faceInfo.size(); bboxId++){
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
	void MTCNN::Padding(int img_w, int img_h){
		for (int i = 0; i < regressed_rects_.size(); i++){
			FaceInfo tempFaceInfo;
			tempFaceInfo = regressed_rects_[i];
			tempFaceInfo.bbox.y2 = (regressed_rects_[i].bbox.y2 >= img_w) ? img_w : regressed_rects_[i].bbox.y2;
			tempFaceInfo.bbox.x2 = (regressed_rects_[i].bbox.x2 >= img_h) ? img_h : regressed_rects_[i].bbox.x2;
			tempFaceInfo.bbox.y1 = (regressed_rects_[i].bbox.y1 < 1) ? 1 : regressed_rects_[i].bbox.y1;
			tempFaceInfo.bbox.x1 = (regressed_rects_[i].bbox.x1 < 1) ? 1 : regressed_rects_[i].bbox.x1;
			regressed_pading_.push_back(tempFaceInfo);
		}
	}

	void MTCNN::GenerateBoundingBox(caffe::Blob<float>* confidence, caffe::Blob<float>* reg,
		float scale, float thresh, int image_width, int image_height){
		int stride = 2;
		int cellSize = 12;

		int curr_feature_map_w_ = std::ceil((image_width - cellSize)*1.0 / stride) + 1;
		int curr_feature_map_h_ = std::ceil((image_height - cellSize)*1.0 / stride) + 1;

		//std::cout << "Feature_map_size:"<< curr_feature_map_w_ <<" "<<curr_feature_map_h_<<std::endl;
		int regOffset = curr_feature_map_w_*curr_feature_map_h_;
		// the first count numbers are confidence of face
		int count = confidence->count() / 2;
		const float* confidence_data = confidence->cpu_data();
		confidence_data += count;
		const float* reg_data = reg->cpu_data();

		condidate_rects_.clear();
		for (int i = 0; i < count; i++){
			if (*(confidence_data + i) >= thresh){
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

	MTCNN::MTCNN(const std::string &proto_model_dir){
	#ifdef CPU_ONLY
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
	#else
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
	#endif
		/* Load the network. */
		PNet_.reset(new caffe::Net<float>((proto_model_dir + "/det1.prototxt"), caffe::TEST));
		PNet_->CopyTrainedLayersFrom(proto_model_dir + "/det1.caffemodel");

		CHECK_EQ(PNet_->num_inputs(), 1) << "Network should have exactly one input.";
		CHECK_EQ(PNet_->num_outputs(), 2) << "Network should have exactly two output, one"
			" is bbox and another is confidence.";

		//RNet_.reset(new Net<float>((proto_model_dir+"/det2.prototxt"), TEST));
		RNet_.reset(new caffe::Net<float>((proto_model_dir + "/det2_input.prototxt"), caffe::TEST));
		RNet_->CopyTrainedLayersFrom(proto_model_dir + "/det2.caffemodel");

		//  CHECK_EQ(RNet_->num_inputs(), 0) << "Network should have exactly one input.";
		//  CHECK_EQ(RNet_->num_outputs(),3) << "Network should have exactly two output, one"
		//                                     " is bbox and another is confidence.";

		ONet_.reset(new caffe::Net<float>((proto_model_dir + "/det3_input.prototxt"), caffe::TEST));
		ONet_->CopyTrainedLayersFrom(proto_model_dir + "/det3.caffemodel");

		//  CHECK_EQ(ONet_->num_inputs(), 1) << "Network should have exactly one input.";
		//  CHECK_EQ(ONet_->num_outputs(),3) << "Network should have exactly three output, one"
		//                                     " is bbox and another is confidence.";

		caffe::Blob<float>* input_layer;
		input_layer = PNet_->input_blobs()[0];
		num_channels_ = input_layer->channels();
		CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
	}

	void MTCNN::WrapInputLayer(std::vector<cv::Mat>* input_channels,
		caffe::Blob<float>* input_layer, const int height, const int width) {
		float* input_data = input_layer->mutable_cpu_data();
		for (int i = 0; i < input_layer->channels(); ++i) {
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}
	}
	// multi test image pass a forward
	void MTCNN::ClassifyFace_MulImage(const std::vector<FaceInfo>& regressed_rects, cv::Mat &sample_single,
		boost::shared_ptr<caffe::Net<float> >& net, double thresh, char netName){
		condidate_rects_.clear();

		int numBox = regressed_rects.size();
		std::vector<caffe::Datum> datum_vector;

		boost::shared_ptr<caffe::MemoryDataLayer<float> > mem_data_layer;
		mem_data_layer = boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(net->layers()[0]);
		int input_width = mem_data_layer->width();
		int input_height = mem_data_layer->height();

		// load crop_img data to datum
		for (int i = 0; i < numBox; i++){
			int pad_top = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
			int pad_left = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
			int pad_right = std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);
			int pad_bottom = std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);

			cv::Mat crop_img = sample_single(cv::Range(regressed_pading_[i].bbox.y1 - 1, regressed_pading_[i].bbox.y2),
				cv::Range(regressed_pading_[i].bbox.x1 - 1, regressed_pading_[i].bbox.x2));
			cv::copyMakeBorder(crop_img, crop_img, pad_left, pad_right, pad_top, pad_bottom, cv::BORDER_CONSTANT, cv::Scalar(0));

			cv::resize(crop_img, crop_img, cv::Size(input_width, input_height), 0, 0, cv::INTER_AREA);
			crop_img = (crop_img - 127.5)*0.0078125;
			caffe::Datum datum;
			CvMatToDatumSignalChannel(crop_img, &datum);
			datum_vector.push_back(datum);
		}
		regressed_pading_.clear();

		/* extract the features and store */
		mem_data_layer->set_batch_size(numBox);
		mem_data_layer->AddDatumVector(datum_vector);
		/* fire the network */
		float no_use_loss = 0;
		net->Forward(&no_use_loss);
		//  CHECK(reinterpret_cast<float*>(crop_img_set.at(0).data) == net->input_blobs()[0]->cpu_data())
		//          << "Input channels are not wrapping the input layer of the network.";

		// return RNet/ONet result
		std::string outPutLayerName = (netName == 'r' ? "conv5-2" : "conv6-2");
	
		const boost::shared_ptr<caffe::Blob<float> > reg = net->blob_by_name(outPutLayerName);
		const boost::shared_ptr<caffe::Blob<float> > confidence = net->blob_by_name("prob1");
		// ONet points_offset != NULL

		const float* confidence_data = confidence->cpu_data();
		const float* reg_data = reg->cpu_data();
		const float* points_data;
		if (netName == 'o')
		{
			std::string pointsLayerName = "conv6-3";
			const boost::shared_ptr<caffe::Blob<float> > points_offset = net->blob_by_name(pointsLayerName);
			points_data = points_offset->cpu_data();
		}

		for (int i = 0; i<numBox; i++){
			if (*(confidence_data + i * 2 + 1) > thresh){
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
				if (netName == 'o'){
					FacePts face_pts;
					float w = faceRect.y2 - faceRect.y1 + 1;
					float h = faceRect.x2 - faceRect.x1 + 1;
					for (int j = 0; j < 5; j++){
						face_pts.y[j] = faceRect.y1 + *(points_data + j + 10 * i) * h - 1;
						face_pts.x[j] = faceRect.x1 + *(points_data + j + 5 + 10 * i) * w - 1;
					}
					faceInfo.facePts = face_pts;
				}
				condidate_rects_.push_back(faceInfo);
			}
		}
	}
	bool MTCNN::CvMatToDatumSignalChannel(const cv::Mat& cv_mat, caffe::Datum* datum){
		if (cv_mat.empty())
			return false;
		int channels = cv_mat.channels();

		datum->set_channels(cv_mat.channels());
		datum->set_height(cv_mat.rows);
		datum->set_width(cv_mat.cols);
		datum->set_label(0);
		datum->clear_data();
		datum->clear_float_data();
		datum->set_encoded(false);

		int datum_height = datum->height();
		int datum_width = datum->width();
		if (channels == 3){
			for (int c = 0; c < channels; c++){
				for (int h = 0; h < datum_height; ++h){
					for (int w = 0; w < datum_width; ++w){
						const float* ptr = cv_mat.ptr<float>(h);
						datum->add_float_data(ptr[w*channels + c]);
					}
				}
			}
		}

		return true;
	}

	int MTCNN::Detect(const cv::Mat& image, std::vector<FaceInfo>& faceInfo){

		// 2~3ms
		// invert to RGB color space and float type
		cv::Mat sample_single, resized;
		image.convertTo(sample_single, CV_32FC3);
		cv::cvtColor(sample_single, sample_single, cv::COLOR_BGR2RGB);
		sample_single = sample_single.t();
 
		int height = image.rows;
		int width = image.cols;
		int minWH = std::min(height, width);
		int factor_count = 0;
		double m = 12. / minSize;
		minWH *= m;
		std::vector<double> scales;
		while (minWH >= 12)
		{
			scales.push_back(m * std::pow(factor, factor_count));
			minWH *= factor;
			++factor_count;
		}
#if SHOW_MTCNN_DEBUG
        std::vector<cv::Mat>Ps;
#endif
		// 11ms main consum
		caffe::Blob<float>* input_layer = PNet_->input_blobs()[0];
		for (int i = 0; i < factor_count; i++)
		{
			double scale = scales[i];
			int ws = std::ceil(height*scale);
			int hs = std::ceil(width*scale);

			// wrap image and normalization using INTER_AREA method
			cv::resize(sample_single, resized, cv::Size(ws, hs), 0, 0, cv::INTER_AREA);
			resized.convertTo(resized, CV_32FC3, 0.0078125, -127.5*0.0078125);

			// input data
			input_layer->Reshape(1, 3, hs, ws);
			PNet_->Reshape();
			std::vector<cv::Mat> input_channels;
			WrapInputLayer(&input_channels, PNet_->input_blobs()[0], hs, ws);
			cv::split(resized, input_channels);

			// check data transform right
			CHECK(reinterpret_cast<float*>(input_channels.at(0).data) == PNet_->input_blobs()[0]->cpu_data())
				<< "Input channels are not wrapping the input layer of the network.";
			PNet_->Forward();

			// return result
			caffe::Blob<float>* reg = PNet_->output_blobs()[0];
			//const float* reg_data = reg->cpu_data();
			caffe::Blob<float>* confidence = PNet_->output_blobs()[1];
			GenerateBoundingBox(confidence, reg, scale, threshold[0], ws, hs);
			std::vector<FaceInfo> bboxes_nms = NonMaximumSuppression(condidate_rects_, 0.5, 'u');
#if SHOW_MTCNN_DEBUG
            cv::Mat pi = image.clone();
            drawDectionResult(pi, bboxes_nms);
            Ps.push_back(pi);
#endif
            total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
#if SHOW_MTCNN_DEBUG
            cv::Mat p = image.clone();
			drawDectionResult(p, total_boxes_);
#endif
		}
#if SHOW_MTCNN_DEBUG
		cv::Mat p = image.clone();
		drawDectionResult(p, total_boxes_);
#endif
		int numBox = total_boxes_.size();
		if (numBox != 0){
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
#if SHOW_MTCNN_DEBUG
			cv::Mat r = image.clone();
			drawDectionResult(r, regressed_rects_);
#endif
			/// three stage
			numBox = regressed_rects_.size();
			if (numBox != 0){
				ClassifyFace_MulImage(regressed_rects_, sample_single, ONet_, threshold[2], 'o');
				regressed_rects_ = BoxRegress(condidate_rects_, 3);
				faceInfo = NonMaximumSuppression(regressed_rects_, 0.7, 'm');
			}
#if SHOW_MTCNN_DEBUG
			cv::Mat o = image.clone();
			drawDectionResult(o, faceInfo);
#endif
		}
		regressed_pading_.clear();
		regressed_rects_.clear();
		condidate_rects_.clear();
		return 0;
	}
}