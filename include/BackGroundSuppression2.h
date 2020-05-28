#ifndef BACKSUPP2_H
#define BACKSUPP2_H

#include <iostream>
#include <algorithm>
#include <vector>
#include "BackGroundSuppression.h"
#include "Collage.h"
#include "tkDNN/utils.h"
#include "tkDNN/CenternetDetection.h"
#include "tkDNN/MobilenetDetection.h"
#include "tkDNN/Yolo3Detection.h"

namespace edge {

class BackGroundSuppression2 : public edge::BackGroundSuppression {
private:
    cv::Scalar colors[256];
    cv::Mat frame_out;
    std::vector<cv::Mat> batchinfo; 
    std::vector<edge::Tile> original_tiles;
    edge::Collage cl;
    cv::Mat backBBres;
    std::vector<cv::Mat> getBatchingFromBackground(const cv::Mat& frame_in);
    void showBoxImage(std::vector<cv::Mat> &box_image);
    
public:
    BackGroundSuppression2(std::string algo, const int width, const int height, const int n_classes);
    ~BackGroundSuppression2() {};
    float overlap(const float p1, const float l1, const float p2, const float l22);
    float boxesIntersection(const tk::dnn::box &b, const edge::Tile &t);
    float boxesUnion(const tk::dnn::box &b, const edge::Tile &t);
    float IoU(const tk::dnn::box &b, const edge::Tile &t);
    void concatDetections(std::vector<tk::dnn::box>& detected);
    void drawTiles(std::vector<cv::Mat>& original_frames, 
                std::vector<std::vector<tk::dnn::box>> batchDetected, std::vector<std::string> classesNames);
    cv::Mat update(const cv::Mat& frame_in);
};
}

#endif /* BACKSUPP2_H */