#ifndef BACKSUPP_H
#define BACKSUPP_H

#include <iostream>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

namespace edge {

class BackGroundSuppression {
private:
    cv::Ptr<cv::BackgroundSubtractor> pBackSub;

public:
    cv::Mat backsup;
    BackGroundSuppression(std::string algo) {
        //create Background Subtractor objects
        if (algo == "MOG2")
            pBackSub = cv::createBackgroundSubtractorMOG2();
        else
            pBackSub = cv::createBackgroundSubtractorKNN();
    };
    ~BackGroundSuppression() {};
    virtual cv::Mat update(const cv::Mat& frame_in) {};

    void getBackgroundSuppression(const cv::Mat& frame_in) {
        pBackSub->apply(frame_in, backsup);
    }
};
}

#endif /* BACKSUPP_H */