

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

bool checkDistance(int pt1, int off1, int pt2, int off2, int th) {
    return ((std::abs(pt1 + off1 - pt2) < th) || (std::abs(pt2 + off2 - pt1) < th));
}

bool checkOverlap(int pt1, int off1, int pt2, int off2, float th) {
    int overlap = std::abs((pt1 + off1) - (pt2 + off2));
    return ((overlap <  th * off1) || (overlap < th * off2));
}

bool checkIntersaction(cv::Rect b1, cv::Rect b2) {
    return (((b1.x < b2.x) && (b1.x + b1.width > b2.x)) || ((b2.x < b1.x) && (b2.x + b2.width > b1.x))) &&
            (((b1.y < b2.y) && (b1.y + b1.height > b2.y)) || ((b2.y < b1.y) && (b2.y + b2.height > b1.y)));
}

cv::Rect boxUnion(cv::Rect b1, cv::Rect b2) {
    cv::Rect box;
    box.x = (b1.x <= b2.x)? b1.x : b2.x;
    box.y = (b1.y <= b2.y)? b1.y : b2.y;
    box.width = (b1.x + b1.width >= b2.x + b2.width)? b1.x + b1.width - box.x : b2.x + b2.width - box.x;
    box.height = (b1.y + b1.height >= b2.y + b2.height)? b1.y + b1.height - box.y : b2.y + b2.height - box.y;
    return box;
}

void box_clustering(std::vector<cv::Rect> &input_box) {
    int distance_th = 50;
    float overlap_th = 0.3;
    bool end = true;
    while(end) {
        end = false;
        //sort boxes for increasing x coordinate
        std::sort(input_box.begin(), input_box.end(), sortByBoxX);
        for(int i = 0; i <input_box.size(); i++) {
            for(int j = i+1; j<input_box.size(); j++) {
                if((checkDistance(input_box.at(i).y, input_box.at(i).height, input_box.at(j).y, input_box.at(j).height, distance_th) &&
                    checkOverlap(input_box.at(i).x, input_box.at(i).width, input_box.at(j).x, input_box.at(j).width, overlap_th)) ||
                    checkIntersaction(input_box.at(i), input_box.at(j))) {
                    input_box.at(i) = boxUnion(input_box.at(i), input_box.at(j));
                    input_box.erase(input_box.begin()+j);
                    i--;
                    end = true;
                    break;
                }
            }
        }

        //sort boxes for increasing y coordinate
        std::sort(input_box.begin(), input_box.end(), sortByBoxY);
        for(int i = 0; i <input_box.size(); i++) {
            for(int j = i+1; j<input_box.size(); j++) {
                if((checkDistance(input_box.at(i).x, input_box.at(i).width, input_box.at(j).x, input_box.at(j).width, distance_th) &&
                    checkOverlap(input_box.at(i).y, input_box.at(i).height, input_box.at(j).y, input_box.at(j).height, overlap_th)) ||
                    checkIntersaction(input_box.at(i), input_box.at(j))) {
                    input_box.at(i) = boxUnion(input_box.at(i), input_box.at(j));
                    input_box.erase(input_box.begin()+j);
                    end = true;
                    i--;
                    break;
                }
            }
        }
    }
}


double iou(const cv::Rect& a, const cv::Rect& b){
    double area1 = double(a.width) * double(a.height);
	double area2 = double(b.width) * double(b.height);

	double x_overlap = std::max(0., std::min(double(a.x + a.width), double(b.x + b.width)) - std::max(double(a.x), double(b.x)));
	double y_overlap = std::max(0., std::min(double(a.y + a.height), double(b.y + b.height)) - std::max(double(a.y), double(b.y)));
	double intersectionArea = x_overlap*y_overlap;
	double unionArea = area1 + area2 - intersectionArea;
    // std::cout<<area1<<" "<<area2<<" "<<intersectionArea<<" "<<unionArea<<std::endl;
    // std::cout<<intersectionArea / unionArea<<std::endl;
	double iou = intersectionArea / unionArea;
    return iou;
}

// void getBatchesFromMovingObjs(const cv::Mat& frame_in, cv::Mat& back_mask, cv::Mat& frame_out, std::vector<cv::Mat>& batches, std::vector<cv::Rect>& or_rects) {    
void getBatchesFromMovingObjs(const cv::Mat& frame_in, cv::Mat& back_mask, cv::Mat& frame_out, std::vector<cv::Mat>& batches, std::vector<cv::Rect>& or_rects, const std::vector<cv::Mat>& tr_batches, const std::vector<cv::Rect>& tr_rects) {    
    //find clusters of points given and image with only moving objects
    std::vector<std::vector<cv::Point>> contours;
    findContours(back_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    // findContours(back_mask.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat frame_vis = cv::Mat(frame_in.rows, frame_in.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat frame_vis2 = cv::Mat(frame_in.rows, frame_in.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    //for each cluster create a box
    frame_out = frame_in.clone();
    std::vector<cv::Rect> box_contours;
    for (int i = 0; i < contours.size(); ++i){
        // Remove small blobs
        if (contours[i].size() < 300) continue;
        //create box
        cv::Rect box = cv::boundingRect(contours[i]);
        box_contours.push_back(box);

        // cv::Mat s = frame_in(box);
        // s.copyTo(frame_vis(box));
    }
    // std::cout<<tr_rects.size()<<std::endl;
    for(int i=0; i<tr_rects.size(); ++i)
        box_contours.push_back(tr_rects[i]);
    box_clustering(box_contours);

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

    // std::cout<<"before NMS: "<<roi_contours.size()<<std::endl;

    std::vector<std::pair<cv::Mat, cv::Rect>> roi_contours_clean;
    std::vector<std::pair<cv::Mat, cv::Rect>> remaining;
    while (roi_contours.size() > 0){
        remaining.clear();
        std::pair<cv::Mat, cv::Rect> cur = roi_contours[0];
        roi_contours_clean.push_back(cur);
        for (size_t j = 1; j < roi_contours.size(); j++){
            if (iou(roi_contours[0].second, roi_contours[j].second) <= 1e-5){
                remaining.push_back(roi_contours[j]);
            }
            // else
                // std::cout<<"iou: "<< iou(roi_contours[0].second, roi_contours[j].second)<<std::endl;
        }
        roi_contours.clear();
        roi_contours = remaining;
    }

    // std::cout<<"after NMS: "<<roi_contours_clean.size()<<std::endl;


    cv::Mat frame_test = frame_out.clone();
    batches.clear();
    or_rects.clear();

    //select only first MAX_BATCHES bigger boxes
    for(int i=0; i<roi_contours_clean.size() && i<MAX_BATCHES; ++i){
        batches.push_back(roi_contours_clean[i].first);
        or_rects.push_back(roi_contours_clean[i].second);
        rectangle(frame_test, roi_contours_clean[i].second, cv::Scalar(0,0,255), 2);
                
    }

    

    // cv::imshow("roi", frame_test);
    // cv::waitKey(1);
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

std::vector<tk::dnn::box> detectionProcess(const edge::DetProcess_t mode, tk::dnn::DetectionNN *detNN, cv::Mat & frame, std::vector<cv::Mat>& batch_dnn_input, edge::Profiler& prof, edge::BackGroundSuppression* bs, const bool first_iteration, cv::Mat *pre_canny, cv::Mat *old_frame, edge::BackGroundSuppression1* bs1, edge::BackGroundSuppression2* bs2, std::vector<tk::dnn::box> *old_det){

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

            std::vector<cv::Mat> tr_batches;
            std::vector<cv::Rect> tr_rects;

            if(old_det != nullptr && old_det->size()){
                
                // cv::Mat frame_out = frame.clone();
                for(auto d: *old_det){
                    // std::cout<<d.x<<" "<<d.y<<" "<<d.x+d.w<<" "<< d.y+d.h<<std::endl;
                    cv::Rect r(cv::Point2i(std::max(0,int(d.x)), std::max(0,int(d.y))), cv::Point2i(std::min(int(d.x+d.w), 1920), std::min(int(d.y+d.h),1080)));
                    tr_rects.push_back(r);
                    tr_batches.push_back(cv::Mat(frame,r));
                    // rectangle(frame_out, r, cv::Scalar(0,0,255), 2);
                }
                // cv::imshow("tr_roi", frame_out);
                // cv::waitKey(1);
            }
            
            //boxes extraction
            prof.tick("extract boxes");
            getBatchesFromMovingObjs(frame, bg_suppressed, fg_boxes, batches, or_rects, tr_batches, tr_rects);
            prof.tock("extract boxes");

            //inference
            prof.tick("inference");
            for(auto b:batches){
                // cv::imshow("b", b);
                // cv::waitKey(0);
                batch_dnn_input.push_back(b.clone());
            }
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
