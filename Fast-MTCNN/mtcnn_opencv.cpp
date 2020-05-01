#include <fstream>
#include <iostream>
#include "mropencv.h"
#if CV_MINOR_VERSION >= 3
#include <opencv2/dnn.hpp>
#endif
using namespace std;
using namespace cv;

const float pnet_stride = 2;
const float pnet_cell_size = 12;
const int pnet_max_detect_num = 5000;
//mean & std
const float mean_val = 127.5f;
const float std_val = 0.0078125f;
//minibatch size
const int step_size = 128;


typedef struct FaceBox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
} FaceBox;
typedef struct FaceInfo {
    float bbox_reg[4];
    float landmark_reg[10];
    float landmark[10];
    FaceBox bbox;
} FaceInfo;



class MTCNN {
public:
    MTCNN(const string& proto_model_dir);
    vector<FaceInfo> Detect_mtcnn(const cv::Mat& img, const int min_size, const float* threshold, const float factor, const int stage);
//protected:
    vector<FaceInfo> ProposalNet(const cv::Mat& img, int min_size, float threshold, float factor);
    vector<FaceInfo> NextStage(const cv::Mat& image, vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold);
    void BBoxRegression(vector<FaceInfo>& bboxes);
    void BBoxPadSquare(vector<FaceInfo>& bboxes, int width, int height);
    void BBoxPad(vector<FaceInfo>& bboxes, int width, int height);
    void GenerateBBox(Mat* confidence, Mat* reg_box, float scale, float thresh);
    std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
    float IoU(float xmin, float ymin, float xmax, float ymax, float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom = false);



//    std::shared_ptr<dnn::Net> PNet_;
//    std::shared_ptr<dnn::Net> ONet_;
//    std::shared_ptr<dnn::Net> RNet_;
public:
    dnn::Net PNet_;
    dnn::Net RNet_;
    dnn::Net ONet_;

    std::vector<FaceInfo> candidate_boxes_;
    std::vector<FaceInfo> total_boxes_;
};


MTCNN::MTCNN(const string& proto_model_dir) {
//    PNet_ = cv::dnn::readNetFromCaffe(proto_model_dir + "/det1.prototxt", proto_model_dir + "/det1_half.caffemodel");
    PNet_ = cv::dnn::readNetFromCaffe(proto_model_dir + "/det1_.prototxt", proto_model_dir + "/det1_.caffemodel");

    RNet_ = cv::dnn::readNetFromCaffe(proto_model_dir + "/det2.prototxt", proto_model_dir + "/det2_half.caffemodel");
    ONet_ = cv::dnn::readNetFromCaffe(proto_model_dir + "/det3-half.prototxt", proto_model_dir + "/det3-half.caffemodel");
}

bool CompareBBox(const FaceInfo & a, const FaceInfo & b) {
    return a.bbox.score > b.bbox.score;
}



float MTCNN::IoU(float xmin, float ymin, float xmax, float ymax,
                 float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom) {
    float iw = std::min(xmax, xmax_) - std::max(xmin, xmin_) + 1;
    float ih = std::min(ymax, ymax_) - std::max(ymin, ymin_) + 1;
    if (iw <= 0 || ih <= 0)
        return 0;
    float s = iw*ih;
    if (is_iom) {
        float ov = s / min((xmax - xmin + 1)*(ymax - ymin + 1), (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1));
        return ov;
    }
    else {
        float ov = s / ((xmax - xmin + 1)*(ymax - ymin + 1) + (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1) - s);
        return ov;
    }
}
void MTCNN::BBoxRegression(vector<FaceInfo>& bboxes) {
//#pragma omp parallel for num_threads(threads_num)
    for (int i = 0; i < bboxes.size(); ++i) {
        FaceBox &bbox = bboxes[i].bbox;
        float *bbox_reg = bboxes[i].bbox_reg;
        float w = bbox.xmax - bbox.xmin + 1;
        float h = bbox.ymax - bbox.ymin + 1;
        bbox.xmin += bbox_reg[0] * w;
        bbox.ymin += bbox_reg[1] * h;
        bbox.xmax += bbox_reg[2] * w;
        bbox.ymax += bbox_reg[3] * h;
    }
}
void MTCNN::BBoxPad(vector<FaceInfo>& bboxes, int width, int height) {
//#pragma omp parallel for num_threads(threads_num)
    for (int i = 0; i < bboxes.size(); ++i) {
        FaceBox &bbox = bboxes[i].bbox;
        bbox.xmin = round(max(bbox.xmin, 0.f));
        bbox.ymin = round(max(bbox.ymin, 0.f));
        bbox.xmax = round(min(bbox.xmax, width - 1.f));
        bbox.ymax = round(min(bbox.ymax, height - 1.f));
    }
}
void MTCNN::BBoxPadSquare(vector<FaceInfo>& bboxes, int width, int height) {
//#pragma omp parallel for num_threads(threads_num)
    for (int i = 0; i < bboxes.size(); ++i) {
        FaceBox &bbox = bboxes[i].bbox;
        float w = bbox.xmax - bbox.xmin + 1;
        float h = bbox.ymax - bbox.ymin + 1;
        float side = h>w ? h : w;
        bbox.xmin = round(max(bbox.xmin + (w - side)*0.5f, 0.f));

        bbox.ymin = round(max(bbox.ymin + (h - side)*0.5f, 0.f));
        bbox.xmax = round(min(bbox.xmin + side - 1, width - 1.f));
        bbox.ymax = round(min(bbox.ymin + side - 1, height - 1.f));
    }
}
void MTCNN::GenerateBBox(Mat* confidence, Mat* reg_box,
                         float scale, float thresh) {
    int feature_map_w_ = confidence->size[3];
    int feature_map_h_ = confidence->size[2];
    int spatical_size = feature_map_w_*feature_map_h_;
//    const float* confidence_data = (float*)(confidence->data + spatical_size);

    std::cout<<confidence->size;
    std::cout<<" "<<scale<<std::endl;


    const float* confidence_data = (float*)(confidence->data);
    confidence_data += spatical_size;

    cv::Mat image(feature_map_h_,feature_map_w_,confidence->type());

    image.data =(unsigned  char*)(confidence_data);
//    cv::imshow("image",image);
//    cv::waitKey(0);




//    std::cout<<confidence_data[0]<<std::endl;

    const float* reg_data = (float*)(reg_box->data);
    candidate_boxes_.clear();
    for (int i = 0; i<spatical_size; i++) {
//        if (confidence_data[i] >= thresh) {
        if (confidence_data[i] <= 1-thresh) {

            int y = i / feature_map_w_;
            int x = i - feature_map_w_ * y;
            FaceInfo faceInfo;
            FaceBox &faceBox = faceInfo.bbox;

            faceBox.xmin = (float)(x * pnet_stride) / scale;
            faceBox.ymin = (float)(y * pnet_stride) / scale;
            faceBox.xmax = (float)(x * pnet_stride + pnet_cell_size - 1.f) / scale;
            faceBox.ymax = (float)(y * pnet_stride + pnet_cell_size - 1.f) / scale;
            faceInfo.bbox_reg[0] = reg_data[i];
            faceInfo.bbox_reg[1] = reg_data[i + spatical_size];
            faceInfo.bbox_reg[2] = reg_data[i + 2 * spatical_size];
            faceInfo.bbox_reg[3] = reg_data[i + 3 * spatical_size];
            faceBox.score = confidence_data[i];
            candidate_boxes_.push_back(faceInfo);
        }
    }
}
std::vector<FaceInfo> MTCNN::NMS(std::vector<FaceInfo>& bboxes,
                                 float thresh, char methodType) {
    std::vector<FaceInfo> bboxes_nms;
    if (bboxes.size() == 0) {
        return bboxes_nms;
    }
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

        FaceBox select_bbox = bboxes[select_idx].bbox;
        float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) * (select_bbox.ymax - select_bbox.ymin + 1));
        float x1 = static_cast<float>(select_bbox.xmin);
        float y1 = static_cast<float>(select_bbox.ymin);
        float x2 = static_cast<float>(select_bbox.xmax);
        float y2 = static_cast<float>(select_bbox.ymax);

        select_idx++;
//#pragma omp parallel for num_threads(threads_num)
        for (int32_t i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            FaceBox & bbox_i = bboxes[i].bbox;
            float x = std::max<float>(x1, static_cast<float>(bbox_i.xmin));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.ymin));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.xmax)) - x + 1;
            float h = std::min<float>(y2, static_cast<float>(bbox_i.ymax)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.xmax - bbox_i.xmin + 1) * (bbox_i.ymax - bbox_i.ymin + 1));
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

vector<FaceInfo> MTCNN::NextStage(const cv::Mat& image, vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold) {
    vector<FaceInfo> res;
    int batch_size = (int)pre_stage_res.size();
    if (batch_size == 0)
        return res;
    Mat* input_layer = nullptr;
    Mat* confidence = nullptr;
    Mat* reg_box = nullptr;
    Mat* reg_landmark = nullptr;

    std::vector< Mat > targets_blobs;



    switch (stage_num) {
        case 2: {
//            input_layer = RNet_->input_blobs()[0];
//            input_layer->Reshape(batch_size, 3, input_h, input_w);
//            RNet_->Reshape();
        }break;
        case 3: {
//            input_layer = ONet_->input_blobs()[0];
//            input_layer->Reshape(batch_size, 3, input_h, input_w);
//            ONet_->Reshape();
        }break;
        default:
            return res;
            break;
    }
//    float * input_data = input_layer->mutable_cpu_data();
    int spatial_size = input_h*input_w;

//#pragma omp parallel for num_threads(threads_num)

    std::vector<cv::Mat> inputs;

    for (int n = 0; n < batch_size; ++n) {
        FaceBox &box = pre_stage_res[n].bbox;
        Mat roi = image(Rect(Point((int)box.xmin, (int)box.ymin), Point((int)box.xmax, (int)box.ymax))).clone();
        resize(roi, roi, Size(input_w, input_h));
        inputs.push_back(roi);
        //resize好的face roi 里面
    }

    //
//    cv::Mat inputBlob = cv::dnn::blobFromImage(resized, std_val,cv::Size(),mean_val);

//    cv::imshow("image",inputs[0]);
//    cv::waitKey(0);


    Mat blob_input = dnn::blobFromImages(inputs, std_val,cv::Size(),cv::Scalar(mean_val,mean_val,mean_val),false);

//    PNet_.setInput(inputBlob, "data");
//    const std::vector< String >  targets_node{"conv4-2","prob1"};
//    std::vector< Mat > targets_blobs;
//    PNet_.forward(targets_blobs,targets_node);

    switch (stage_num) {
        case 2: {
            RNet_.setInput(blob_input, "data");
            const std::vector< String >  targets_node{"conv5-2","prob1"};
            RNet_.forward(targets_blobs,targets_node);
            confidence = &targets_blobs[1];
            reg_box = &targets_blobs[0];

            float* confidence_data = (float*)confidence->data;
        }break;
        case 3: {

            ONet_.setInput(blob_input, "data");
            const std::vector< String >  targets_node{"conv6-2","conv6-3","prob1"};
            ONet_.forward(targets_blobs,targets_node);
            reg_box = &targets_blobs[0];
            reg_landmark = &targets_blobs[1];
            confidence = &targets_blobs[2];

        }break;
    }


    const float* confidence_data = (float*)confidence->data;
//    std::cout<<"confidence_data[0] "<<confidence_data[0]<<std::endl;

    const float* reg_data = (float*)reg_box->data;
    const float* landmark_data = nullptr;
    if (reg_landmark) {
        landmark_data = (float*)reg_landmark->data;
    }
    for (int k = 0; k < batch_size; ++k) {
        if (confidence_data[2 * k + 1] >= threshold) {
            FaceInfo info;
            info.bbox.score = confidence_data[2 * k + 1];
            info.bbox.xmin = pre_stage_res[k].bbox.xmin;
            info.bbox.ymin = pre_stage_res[k].bbox.ymin;
            info.bbox.xmax = pre_stage_res[k].bbox.xmax;
            info.bbox.ymax = pre_stage_res[k].bbox.ymax;
            for (int i = 0; i < 4; ++i) {
                info.bbox_reg[i] = reg_data[4 * k + i];
            }
            if (reg_landmark) {
                float w = info.bbox.xmax - info.bbox.xmin + 1.f;
                float h = info.bbox.ymax - info.bbox.ymin + 1.f;
                for (int i = 0; i < 5; ++i){
                    info.landmark[2 * i] = landmark_data[10 * k + 2 * i] * w + info.bbox.xmin;
                    info.landmark[2 * i + 1] = landmark_data[10 * k + 2 * i + 1] * h + info.bbox.ymin;
                }
            }
            res.push_back(info);
        }
    }
    return res;
}

vector<FaceInfo> MTCNN::ProposalNet(const cv::Mat& img, int minSize, float threshold, float factor) {
    cv::Mat  resized;
    int width = img.cols;
    int height = img.rows;
    float scale = 12.f / minSize;
    float minWH = std::min(height, width) *scale;
    std::vector<float> scales;
    while (minWH >= 12) {
        scales.push_back(scale);
        minWH *= factor;
        scale *= factor;
    }

//    Mat* input_layer = PNet_->input_blobs()[0];
    total_boxes_.clear();
    for (int i = 0; i < scales.size(); i++) {
        int ws = (int)std::ceil(width*scales[i]);
        int hs = (int)std::ceil(height*scales[i]);
        cv::resize(img, resized, cv::Size(ws, hs), 0, 0, cv::INTER_LINEAR);
//
//        input_layer->Reshape(1, 3, hs, ws);
//        PNet_->Reshape();
//
//        float * input_data = input_layer->mutable_cpu_data();
//        cv::Vec3b * img_data = (cv::Vec3b *)resized.data;
//        int spatial_size = ws* hs;
//        for (int k = 0; k < spatial_size; ++k) {
//            input_data[k] = float((img_data[k][0] - mean_val)* std_val);
//            input_data[k + spatial_size] = float((img_data[k][1] - mean_val) * std_val);
//            input_data[k + 2 * spatial_size] = float((img_data[k][2] - mean_val) * std_val);
//        }


        cv::Mat inputBlob = cv::dnn::blobFromImage(resized, 1/255.0,cv::Size(),cv::Scalar(0,0,0),false);

        float* c = (float*)inputBlob.data;
        PNet_.setInput(inputBlob, "data");
        const std::vector< cv::String >  targets_node{"conv4-2","prob1"};
        std::vector< cv::Mat > targets_blobs;
        PNet_.forward(targets_blobs,targets_node);

        cv::Mat prob = targets_blobs[1]
        ;
        cv::Mat reg = targets_blobs[0];
        GenerateBBox(&prob, &reg, scales[i], threshold);
//
        std::vector<FaceInfo> bboxes_nms = NMS(candidate_boxes_, 0.5, 'u');
        if (bboxes_nms.size()>0) {
            total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
        }
    }
    int num_box = (int)total_boxes_.size();
//    std::cout<<num_box<<std::endl;

    vector<FaceInfo> res_boxes;
    if (num_box != 0) {
        res_boxes = NMS(total_boxes_, 0.7f, 'u');
        BBoxRegression(res_boxes);
        BBoxPadSquare(res_boxes, width, height);
    }
    return res_boxes;
}

vector<FaceInfo> MTCNN::Detect_mtcnn(const cv::Mat& image, const int minSize, const float* threshold, const float factor, const int stage) {
    vector<FaceInfo> pnet_res;
    vector<FaceInfo> rnet_res;
    vector<FaceInfo> onet_res;
    if (stage >= 1){
        pnet_res = ProposalNet(image, minSize, threshold[0], factor);
    }
    if (stage >= 2 && pnet_res.size()>0){
        if (pnet_max_detect_num < (int)pnet_res.size()){
            pnet_res.resize(pnet_max_detect_num);
        }
        int num = (int)pnet_res.size();
        int size = (int)ceil(1.f*num / step_size);
        for (int iter = 0; iter < size; ++iter){
            int start = iter*step_size;
            int end = min(start + step_size, num);
            vector<FaceInfo> input(pnet_res.begin() + start, pnet_res.begin() + end);
            vector<FaceInfo> res = NextStage(image, input, 24, 24, 2, threshold[1]);
            rnet_res.insert(rnet_res.end(), res.begin(), res.end());
        }
        rnet_res = NMS(rnet_res, 0.4f, 'm');
        BBoxRegression(rnet_res);
        BBoxPadSquare(rnet_res, image.cols, image.rows);

    }
    if (stage >= 3 && rnet_res.size()>0){
        int num = (int)rnet_res.size();
        int size = (int)ceil(1.f*num / step_size);
        for (int iter = 0; iter < size; ++iter){
            int start = iter*step_size;
            int end = min(start + step_size, num);
            vector<FaceInfo> input(rnet_res.begin() + start, rnet_res.begin() + end);
            vector<FaceInfo> res = NextStage(image, input, 48, 48, 3, threshold[2]);
            onet_res.insert(onet_res.end(), res.begin(), res.end());
        }
        BBoxRegression(onet_res);
        onet_res = NMS(onet_res, 0.4f, 'm');
        BBoxPad(onet_res, image.cols, image.rows);

    }
    if (stage == 1){
        return pnet_res;
    }
    else if (stage == 2){
        return rnet_res;
    }
    else if (stage == 3){
        return onet_res;
    }
    else{
        return onet_res;
    }
}

int main(int argc, char **argv)
{

    MTCNN detector("model");
    string name_list[1] = {
		"../imgs/test.jpg",
	};

//	MTCNN detector("./model");
	float factor = 0.709f;
	float threshold[3] = { 0.7f, 0.6f, 0.6f };
	int minSize = 12;
	for (int n = 0; n < 1;++n){
		cv::Mat image = cv::imread(name_list[n], 1);

        for(int i = 0 ; i < 10 ; i ++) {
            double t = (double) cv::getTickCount();
            vector<FaceInfo> faceInfo = detector.Detect_mtcnn(image, minSize, threshold, factor, 3);
            std::cout << name_list[n] << " time," << (double) (cv::getTickCount() - t) / cv::getTickFrequency() << "s"
                      << std::endl;

            for (int i = 0; i < faceInfo.size(); i++) {
                int x = (int) faceInfo[i].bbox.xmin;
                int y = (int) faceInfo[i].bbox.ymin;
                int w = (int) (faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
                int h = (int) (faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
                cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
            }
            cv::imwrite("test.png", image);
            cv::imshow("image", image);
            cv::waitKey(0);

        }


    }
 
	 
	return 1;
}
