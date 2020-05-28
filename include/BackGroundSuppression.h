#ifndef BACKSUPP_H
#define BACKSUPP_H

#include <iostream>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudabgsegm.hpp>

namespace edge {

class BackGroundSuppression {
private:
    cv::Ptr<cv::BackgroundSubtractor> pBackSub;
    cv::cuda::GpuMat in, out;
    bool onGPU;

public:
    cv::Mat backsup;
    BackGroundSuppression(const bool on_gpu=true) {
        onGPU = on_gpu;
        //create Background Subtractor objects
        if (on_gpu)
            pBackSub = cv::cuda::createBackgroundSubtractorMOG2();
        else
            pBackSub = cv::createBackgroundSubtractorKNN();
    };
    ~BackGroundSuppression() {};
    virtual cv::Mat update(const cv::Mat& frame_in) {};

    cv::Mat getBackgroundSuppression(const cv::Mat& frame_in) {
        if (onGPU){
            in.upload(frame_in);
            pBackSub->apply(in, out);
            out.download(backsup);
        }
        else
            pBackSub->apply(frame_in, backsup);
        return backsup;
    }
};
}

#endif /* BACKSUPP_H */