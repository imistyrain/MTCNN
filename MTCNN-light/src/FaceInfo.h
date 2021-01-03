#pragma once
#include <cstdint>
namespace facedetecion
{
    enum FaceDetectionResult{
        OK,
        MODELS_NOT_FOUND,
        TOO_LESS_BBOXES_PNET,
        TOO_LESS_BBOXES_RNET,
        TOO_LESS_BBOXES_ONET,
    };
    typedef struct Rect {
        int32_t x;
        int32_t y;
        int32_t width;
        int32_t height;
    } Rect;
    typedef struct FaceInfo {
        Rect bbox;
        double roll;
        double pitch;
        double yaw;
        double score;
    } FaceInfo;
}