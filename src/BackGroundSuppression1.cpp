#include "BackGroundSuppression1.h"


namespace edge {

BackGroundSuppression1::BackGroundSuppression1(const bool on_gpu, const int width, const int height) : BackGroundSuppression(on_gpu){
    frame_dark = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
}

cv::Mat BackGroundSuppression1::update(const cv::Mat& frame_in) {
    getBackgroundSuppression(frame_in);
    std::vector<std::vector<cv::Point>> contours;
    findContours(backsup.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    int th_blob_size = 100;
    frame_out = frame_dark.clone();
    // std::cout<<"frame_back: "<<frame_out.cols<<" * "<<frame_out.rows<<std::endl;
    for (int i = 0; i < contours.size(); ++i)
    {
        // Remove small blobs
        if (contours[i].size() < th_blob_size)
        {
            continue;
        }
        cv::Rect box = cv::boundingRect(contours[i]);
        cv::Mat s = frame_in(box);
        // std::cout<<"s: "<<s.cols<<" * "<<s.rows<<std::endl;
        s.copyTo(frame_out(box));
    }
    return frame_out;
}
}