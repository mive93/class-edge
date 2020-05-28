#ifndef ROI_H
#define ROI_H

#include <iostream>

#include "tkDNN/CenternetDetection.h"
#include "tkDNN/MobilenetDetection.h"
#include "tkDNN/Yolo3Detection.h"

#include "undistort.h"
#include "configuration.h"
#include "Profiler.h"
#include "BackGroundSuppression1.h"
#include "BackGroundSuppression2.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#define MAX_BATCHES 32

namespace edge{
enum DetProcess_t { FULL_IMG,       //default, feeding the netwotk with full image
                    BS_BATCHES,     //feeding the network with batches got from background suppression
                    CANNY_BATCHES,  //feeding the network with batches got from frame disparity + canny
                    OF_BATCHES,     //feeding the network with batches got from frame disparity
                    FULL_IMG_BS,    //feeding the network with black full image and only colored foreground objects
                    COLLAGE         //feeding the network with collage of batches got from background suppression
                    };
}

cv::Mat getDisparityCanny(const cv::Mat& frame,cv::Mat& pre_canny);
cv::Mat getDisparityOpticalFlow(const cv::Mat& frame, cv::Mat& old_frame);

bool sortByRoiSize(const std::pair<cv::Mat, cv::Rect> &a, const std::pair<cv::Mat, cv::Rect> &b);
void getBatchesFromMovingObjs(  const cv::Mat& frame_in, cv::Mat& back_mask, cv::Mat& frame_out, 
                                std::vector<cv::Mat>& batches, std::vector<cv::Rect>& or_rects);

std::vector<tk::dnn::box> concatDetections( const std::vector<std::vector<tk::dnn::box>>& batchDetected, 
                                            const std::vector<cv::Rect>& or_rects);

std::vector<tk::dnn::box> detectionProcess( const edge::DetProcess_t mode, tk::dnn::DetectionNN *detNN, cv::Mat & frame, 
                                            std::vector<cv::Mat>& batch_dnn_input, edge::Profiler& prof, 
                                            edge::BackGroundSuppression* bs=nullptr, const bool first_iteration=false, 
                                            cv::Mat *pre_canny=nullptr, cv::Mat *old_frame=nullptr, 
                                            edge::BackGroundSuppression1* bs1=nullptr, edge::BackGroundSuppression2* bs2=nullptr);

#endif /*ROI_H*/