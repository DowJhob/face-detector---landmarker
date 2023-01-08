#ifndef _PEPPA_PIG_FACE_LANDMARK_H_
#define _PEPPA_PIG_FACE_LANDMARK_H_

#include <vector>

#include "opencv2/core.hpp"
#include "../landmarker.h"

#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"

namespace mirror {
    class Peppa_Pig_Face_Landmarker : public Landmarker {
    public:
        Peppa_Pig_Face_Landmarker();
        ~Peppa_Pig_Face_Landmarker();
        int Init(const char* model_path);
        int ExtractKeypoints(const cv::Mat& img_src, const cv::Rect& face, std::vector<cv::Point2f>* keypoints);
        int min_face = 20;

    private:
        bool initialized_;
        std::shared_ptr<MNN::CV::ImageProcess> pretreat_ = nullptr;
        std::shared_ptr<MNN::Interpreter> pig_interpreter_ = nullptr;
        MNN::Session* pig_sess_ = nullptr;
        MNN::Tensor* input_tensor_ = nullptr;

        const cv::Size inputSize_ = cv::Size(128, 128);
        const float meanVals_[3] = { 127.5f, 127.5f, 127.5f };
        const float normVals_[3] = { 0.0078125f, 0.0078125f, 0.0078125f };

    };


}




#endif // !_PEPPA_PIG_FACE_LANDMARK_H_