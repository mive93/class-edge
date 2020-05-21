#ifndef BACKSUPP1_H
#define BACKSUPP1_H

#include "BackGroundSuppression.h"

namespace edge {

class BackGroundSuppression1 : public edge::BackGroundSuppression {
private:
    cv::Mat frame_dark;
    cv::Mat frame_out;
public:
    BackGroundSuppression1(std::string algo, const int width, const int height);
    ~BackGroundSuppression1() {};
    cv::Mat update(const cv::Mat& frame_in);
};
}

#endif /* BACKSUPP1_H */