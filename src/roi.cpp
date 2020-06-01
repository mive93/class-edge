

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

bool sortByBoxX(const cv::Rect &a, const cv::Rect &b) { 
    return (a.x < b.x); 
} 

bool sortByBoxY(const cv::Rect &a, const cv::Rect &b) { 
    return (a.y < b.y);
} 

std::vector<cv::Rect> box_clustering(std::vector<cv::Rect> &input_box) {
    std::vector<cv::Rect> output_box;
    std::vector<cv::Rect> x_box;
    std::vector<int> id_eaten;
    int distance_th = 5;
    int overlap_th = 0.1;
    bool union_flag;
    //sort boxes for increasing x coordinate
    std::sort(input_box.begin(), input_box.end(), sortByBoxX);
    for(int i = 0; i <input_box.size(); i++) {
        union_flag = false;
        // already seen it
        if(std::count(id_eaten.begin(), id_eaten.end(), i))
            continue;
        for(int j = i; j<input_box.size(); j++) {
            // no overlap (element are sorted)
            if(input_box.at(i).x + input_box.at(i).width < input_box.at(j).x)
                break;
            if(std::abs(input_box.at(i).x + input_box.at(i).height - input_box.at(j).x) < distance_th &&
               std::abs((input_box.at(i).x + input_box.at(i).width) - (input_box.at(j).x + input_box.at(j).width) < 
                        overlap_th * input_box.at(i).width)) {
                union_flag = true;
                id_eaten.push_back(j);
                cv::Rect box;
                box.x = std::min(input_box.at(i).x, input_box.at(j).x);
                box.y = std::min(input_box.at(i).y, input_box.at(j).y);
                box.width = std::max(input_box.at(i).x + input_box.at(i).width, input_box.at(j).x + input_box.at(j).width) - box.x;
                box.x = std::max(input_box.at(i).y + input_box.at(i).height, input_box.at(j).y + input_box.at(j).height) - box.y;
                x_box.push_back(box);
            }
        }
        if(!union_flag)
            x_box.push_back(input_box.at(i));
    }
    id_eaten.clear();
    //sort boxes for increasing y coordinate
    std::sort(x_box.begin(), x_box.end(), sortByBoxY);
    for(int i = 0; i <x_box.size(); i++) {
        union_flag = false;
        // already seen it
        if(std::count(id_eaten.begin(), id_eaten.end(), i))
            continue;
        for(int j = i; j<x_box.size(); j++) {
            // no overlap (element are sorted)
            if(x_box.at(i).y + x_box.at(i).height < x_box.at(j).y)
                break;
            if(std::abs(x_box.at(i).y + x_box.at(i).width - x_box.at(j).y) < distance_th &&
               std::abs((x_box.at(i).y + x_box.at(i).height) - (x_box.at(j).y + x_box.at(j).height) < 
                        overlap_th * x_box.at(i).height)) {
                union_flag = true;
                id_eaten.push_back(j);
                cv::Rect box;
                box.x = std::min(x_box.at(i).x, x_box.at(j).x);
                box.y = std::min(x_box.at(i).y, x_box.at(j).y);
                box.width = std::max(x_box.at(i).x + x_box.at(i).width, x_box.at(j).x + x_box.at(j).width) - box.x;
                box.x = std::max(x_box.at(i).y + x_box.at(i).height, x_box.at(j).y + x_box.at(j).height) - box.y;
                output_box.push_back(box);
            }
        }
        if(!union_flag)
            output_box.push_back(input_box.at(i));
    }

    return output_box;

}

void getBatchesFromMovingObjs(const cv::Mat& frame_in, cv::Mat& back_mask, cv::Mat& frame_out, std::vector<cv::Mat>& batches, std::vector<cv::Rect>& or_rects) {    
    //find clusters of points given and image with only moving objects
    std::vector<std::vector<cv::Point>> contours;
    findContours(back_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    //for each cluster create a box
    frame_out = frame_in.clone();
    std::vector<cv::Rect> box_contours;
    for (int i = 0; i < contours.size(); ++i){
        // Remove small blobs
        if (contours[i].size() < 100) continue;
        //create box
        cv::Rect box = cv::boundingRect(contours[i]);
        box_contours.push_back(box);
    }
    box_contours = box_clustering(box_contours);

    std::vector<std::pair<cv::Mat, cv::Rect>> roi_contours;
    for (auto box : box_contours) {
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
        case edge::FULL_IMG: {
            //inference
            prof.tick("inference");
            batch_dnn_input.push_back(frame.clone());
            detNN->update(batch_dnn_input, batch_dnn_input.size());
            prof.tock("inference");
            
            return detNN->detected;
        }
        case edge::COLLAGE: {
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
        }
        case edge::FULL_IMG_BS: {
            //background suppression
            prof.tick("backgroundsuppression1");
            if(bs1 == nullptr)
                FatalError("bs1 needed");
            bg_suppressed = bs1->update(frame);
            prof.tock("backgroundsuppression1");
            std::cout<<"bs_suppression --- "<<bg_suppressed.size()<<std::endl;
            //inference
            prof.tick("inference");
            batch_dnn_input.push_back(bg_suppressed.clone());
            detNN->update(batch_dnn_input, batch_dnn_input.size());
            prof.tock("inference");

            //draw boxes
            return detNN->detected;
        }
        case edge::FULL_IMG_BS_ONLY_RGB: {
            cv::Mat mask;
            //background suppression
            prof.tick("backgroundsuppression_rgb");
            if(bs == nullptr)
                FatalError("bs needed");
            mask = bs->getBackgroundSuppression(frame);
            // cv::bitwise_not(frame, bg_suppressed, aus);
            cv::copyTo(frame, bg_suppressed, mask);
            // cv::imshow("bs", bg_suppressed);
            // cv::waitKey(1);
            prof.tock("backgroundsuppression_rgb");

            //inference
            prof.tick("inference");
            batch_dnn_input.push_back(bg_suppressed.clone());
            detNN->update(batch_dnn_input, batch_dnn_input.size());
            prof.tock("inference");

            //draw boxes
            return detNN->detected;
        }
        default:
            break;
    }
    std::vector<tk::dnn::box> detected;
    return detected;
}
