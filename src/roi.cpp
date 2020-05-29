

#include "roi.h"

cv::Mat getDisparityCanny(const cv::Mat& frame,cv::Mat& pre_canny){
    cv::Mat canny;
    cv::Canny(frame, canny, 100, 100 * 2);
    cv::Mat disparity = canny - pre_canny;
    pre_canny = canny.clone();

    int dilation_type = cv::MORPH_RECT;
    int dilation_size = 1;
    cv::Mat element = cv::getStructuringElement( dilation_type,
                            cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                            cv::Point( dilation_size, dilation_size ) );
    
    cv::Mat disparity_dilated;
    dilate( disparity, disparity_dilated, element);
    return disparity_dilated;
}

cv::Mat getDisparityOpticalFlow(const cv::Mat& frame, cv::Mat& old_frame){
    cv::Mat disparity = old_frame.clone() - frame.clone();
    old_frame = frame.clone();

    cv::Mat disparity_grey;
    cv::cvtColor(disparity, disparity_grey, cv::COLOR_BGR2GRAY);
    return disparity_grey;
}

bool sortByRoiSize(const std::pair<cv::Mat, cv::Rect> &a, const std::pair<cv::Mat, cv::Rect> &b) { 
    return (a.first.rows*a.first.cols > b.first.rows*b.first.cols); 
} 

void getBatchesFromMovingObjs(const cv::Mat& frame_in, cv::Mat& back_mask, cv::Mat& frame_out, std::vector<cv::Mat>& batches, std::vector<cv::Rect>& or_rects) {    
    //find clusters of points given and image with only moving objects
    std::vector<std::vector<cv::Point>> contours;
    findContours(back_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    //for each cluster create a box
    frame_out = frame_in.clone();
    std::vector<std::pair<cv::Mat, cv::Rect>> roi_contours;
    for (int i = 0; i < contours.size(); ++i){
        // Remove small blobs
        if (contours[i].size() < 100) continue;
        //create box
        cv::Rect box = cv::boundingRect(contours[i]);
        //extract box from original image
        cv::Mat roi = cv::Mat(frame_in,box);
        roi_contours.push_back(std::pair<cv::Mat, cv::Rect>(roi, box));
        //draw box on the original image
        rectangle(frame_out, box, cv::Scalar(0,0,255), 2);
    }
    //sort boxes for decreasing size
    std::sort(roi_contours.begin(), roi_contours.end(), sortByRoiSize); 

    //select only first MAX_BATCHES bigger boxes
    for(int i=0; i<roi_contours.size() && i<MAX_BATCHES; ++i){
        batches.push_back(roi_contours[i].first);
        or_rects.push_back(roi_contours[i].second);
    }
}

std::vector<tk::dnn::box> concatDetections(const std::vector<std::vector<tk::dnn::box>>& batchDetected, const std::vector<cv::Rect>& or_rects){
    std::vector<tk::dnn::box> detected;
    for(int bi=0; bi<batchDetected.size(); ++bi){
        for(int i=0; i<batchDetected[bi].size(); i++) { 
            tk::dnn::box b;
            b.x         = batchDetected[bi][i].x+or_rects[bi].x;
            b.y         = batchDetected[bi][i].y+or_rects[bi].y;
            b.w         = batchDetected[bi][i].w;
            b.h         = batchDetected[bi][i].h;
            b.cl        = batchDetected[bi][i].cl;
            b.prob      = batchDetected[bi][i].prob;
            b.probs     = batchDetected[bi][i].probs;
            detected.push_back(b);
        }
    }
    return detected;
}

std::vector<tk::dnn::box> detectionProcess(const edge::DetProcess_t mode, tk::dnn::DetectionNN *detNN, cv::Mat & frame, std::vector<cv::Mat>& batch_dnn_input, edge::Profiler& prof, edge::BackGroundSuppression* bs, const bool first_iteration, cv::Mat *pre_canny, cv::Mat *old_frame, edge::BackGroundSuppression1* bs1, edge::BackGroundSuppression2* bs2){

    batch_dnn_input.clear();
    cv::Mat bg_suppressed;

    switch (mode){
        //background suppression    
        case edge::BS_BATCHES:
            prof.tick("background");
            bg_suppressed = bs->getBackgroundSuppression(frame);
            prof.tock("background");
            break;

        //canny
        case edge::CANNY_BATCHES:
            if(pre_canny == nullptr)
                FatalError("Pre canny needed");
            if(first_iteration)
                cv::Canny(frame, *pre_canny, 100, 100 * 2);    
            prof.tick("canny");
            bg_suppressed = getDisparityCanny(frame, *pre_canny);
            prof.tock("canny");
            break;

        //frame disparity
        case edge::OF_BATCHES:
            if(old_frame == nullptr)
                FatalError("old_frame needed");
            if(first_iteration)
                *old_frame = frame.clone();
            prof.tick("disparity");
            bg_suppressed = getDisparityOpticalFlow(frame, *old_frame);
            prof.tock("disparity");
            break;
        }
        
        switch (mode){
        case edge::BS_BATCHES: case edge::OF_BATCHES: case edge::CANNY_BATCHES:{
            cv::Mat fg_boxes;
            std::vector<cv::Mat> batches;
            std::vector<cv::Rect> or_rects;
            
            //boxes extraction
            prof.tick("extract boxes");
            getBatchesFromMovingObjs(frame, bg_suppressed, fg_boxes, batches, or_rects);
            prof.tock("extract boxes");

            //inference
            prof.tick("inference");
            for(auto b:batches)
                batch_dnn_input.push_back(b.clone());
            detNN->update(batch_dnn_input, batch_dnn_input.size());
            prof.tock("inference");
            
            //merge detections
            prof.tick("convert dets");
            auto detected = concatDetections(detNN->batchDetected, or_rects);
            prof.tock("convert dets");
            return detected;
        }
        case edge::FULL_IMG:
            //inference
            prof.tick("inference");
            batch_dnn_input.push_back(frame.clone());
            detNN->update(batch_dnn_input, batch_dnn_input.size());
            prof.tock("inference");
            
            return detNN->detected;
        case edge::COLLAGE:
            //background suppression
            prof.tick("backgroundsuppression2");
            if(bs2 == nullptr)
                FatalError("bs2 needed");
            bg_suppressed = bs2->update(frame);
            prof.tock("backgroundsuppression2");
            
            //inference
            prof.tick("inference");
            batch_dnn_input.push_back(bg_suppressed.clone());
            detNN->update(batch_dnn_input, batch_dnn_input.size());
            prof.tock("inference");

            //draw boxes
            prof.tick("convert dets");
            bs2->concatDetections(detNN->detected);
            prof.tock("convert dets");
            return detNN->detected;
        case edge::FULL_IMG_BS:
            //background suppression
            prof.tick("backgroundsuppression1");
            if(bs1 == nullptr)
                FatalError("bs1 needed");
            bg_suppressed = bs1->update(frame);
            prof.tock("backgroundsuppression1");

            //inference
            prof.tick("inference");
            batch_dnn_input.push_back(bg_suppressed.clone());
            detNN->update(batch_dnn_input, batch_dnn_input.size());
            prof.tock("inference");

            //draw boxes
            return detNN->detected;
        default:
            break;
    }
    std::vector<tk::dnn::box> detected;
    return detected;
}
