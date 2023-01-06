#include "Peppa_Pig_Face_Landmark.h"

#include <iostream>
#include <string>

#include "opencv2/imgproc.hpp"

namespace mirror {
    Peppa_Pig_Face_Landmarker::Peppa_Pig_Face_Landmarker()
    {
        initialized_ = false;
    }

    Peppa_Pig_Face_Landmarker::~Peppa_Pig_Face_Landmarker()
    {
        pig_interpreter_->releaseModel();
        pig_interpreter_->releaseSession(pig_sess_);
    }

    int Peppa_Pig_Face_Landmarker::Init(const char* model_path) {
        std::cout << "start init." << std::endl;
        std::string model_file = std::string(model_path) + "/Peppa_Pig_Face_Landmark/kps.mnn";
        //std::string model_file = std::string(model_path) + "/Peppa_Pig_Face_Landmark/kps_68.mnn";
        pig_interpreter_ = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
        if (nullptr == pig_interpreter_) {
            std::cout << "load model failed." << std::endl;
            return 10000;
        }

        // create session
        MNN::ScheduleConfig schedule_config;
        schedule_config.type = MNN_FORWARD_CPU;
        schedule_config.numThread = 1;
        MNN::BackendConfig backend_config;
        backend_config.memory = MNN::BackendConfig::Memory_Normal;
        backend_config.power = MNN::BackendConfig::Power_High;
        backend_config.precision = MNN::BackendConfig::Precision_High;
        schedule_config.backendConfig = &backend_config;
        pig_sess_ = pig_interpreter_->createSession(schedule_config);
        input_tensor_ = pig_interpreter_->getSessionInput(pig_sess_, nullptr);

        MNN::CV::Matrix trans;
        trans.setScale(1.0f, 1.0f);
        MNN::CV::ImageProcess::Config img_config;
        img_config.filterType = MNN::CV::BICUBIC;
        ::memcpy(img_config.mean, meanVals_, sizeof(meanVals_));
        ::memcpy(img_config.normal, normVals_, sizeof(normVals_));
        img_config.sourceFormat = MNN::CV::BGR;
        img_config.destFormat = MNN::CV::RGB;
        pretreat_ = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(img_config));
        pretreat_->setMatrix(trans);

        initialized_ = true;

        std::cout << "end init." << std::endl;

        return 0;
    }

    int Peppa_Pig_Face_Landmarker::ExtractKeypoints(const cv::Mat& img_src, const cv::Rect& face, std::vector<cv::Point2f>* keypoints)
    {
        std::cout << "start extract keypoints." << std::endl;
        keypoints->clear();
        if (!initialized_) {
            std::cout << "model uninitialized." << std::endl;
            return 10000;
        }
        if (img_src.empty()) {
            std::cout << "input empty." << std::endl;
            return 10001;
        }
        cv::Mat img_face = img_src(face).clone();
        int width = img_face.cols;
        int height = img_face.rows;
        cv::Mat img_resized;
        cv::resize(img_face, img_resized, inputSize_);
        //pig_interpreter_->resizeTensor(input_tensor_, 1, 3, height, width);
        pretreat_->convert(img_resized.data, inputSize_.width, inputSize_.height, 0, input_tensor_);

        //pig_interpreter_->resizeSession(pig_sess_);



        // run session
        pig_interpreter_->runSession(pig_sess_);

        // get output
        std::string output_name = "";
        auto output_landmark = pig_interpreter_->getSessionOutput(pig_sess_, nullptr);
        MNN::Tensor landmark_tensor(output_landmark, output_landmark->getDimensionType());
        output_landmark->copyToHostTensor(&landmark_tensor);

        for (int i = 0; i < 106; ++i) {
            cv::Point2f curr_pt(landmark_tensor.host<float>()[2 * i + 0] * width + face.x,
                landmark_tensor.host<float>()[2 * i + 1] * height + face.y);
            keypoints->push_back(curr_pt);
        }

        std::cout << "end extract keypoints." << std::endl;

        return 0;
    }

    /*
    void Peppa_Pig_Face_Landmark::postprocess(std::vector<cv::Point2f>* landmark, detail)
    {

        ##recorver, and grouped as[68, 2]

            # landmark[:, 0] = landmark[:, 0] * w + bbox[0] - add
            # landmark[:, 1] = landmark[:, 1] * h + bbox[1] - add
            landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3] - detail[4]
            landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2] - detail[4]

            return landmark
    }
    */
}