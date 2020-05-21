#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>

#include "tkDNN/CenternetDetection.h"
#include "tkDNN/MobilenetDetection.h"
#include "tkDNN/Yolo3Detection.h"
#include "undistort.h"
#include "configuration.h"
#include "Profiler.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "BackGroundSuppression1.h"
#include "BackGroundSuppression2.h"

#define MAX_BATCHES 32

cv::Mat getDisparityCanny(const cv::Mat& frame, const bool first_iteration, cv::Mat& canny,cv::Mat& pre_canny,cv::Mat& canny_RGB,cv::Mat& pre_canny_RGB){
    cv::Canny(frame, canny, 100, 100 * 2);
    cv::Mat disparity = canny.clone();
    if (!first_iteration){
        cv::cvtColor(pre_canny, pre_canny_RGB, cv::COLOR_GRAY2RGB);
        cv::cvtColor(canny, canny_RGB, cv::COLOR_GRAY2RGB);
        disparity = canny_RGB.clone() - pre_canny_RGB.clone();
    }
    pre_canny = canny.clone();

    return disparity;
}

cv::Mat getDisparityOpticalFlow(const cv::Mat& frame, cv::Mat& old_frame, const bool& first_iteration){
    if(first_iteration){
        old_frame = frame.clone();
        old_frame.setTo(cv::Scalar(0,0,0));
    }

    cv::Mat disparity = old_frame.clone() - frame.clone();
    old_frame = frame.clone();

    cv::Mat disparity_grey;
    cv::cvtColor(disparity, disparity_grey, cv::COLOR_BGR2GRAY);
    return disparity_grey;
}

void suppressBackground(const cv::Mat& frame_in, cv::Mat& frame_out, cv::Ptr<cv::BackgroundSubtractor> bg_subtractor, const bool on_gpu=true) {
    if (on_gpu){
        cv::cuda::GpuMat in, out;
        in.upload(frame_in);
        bg_subtractor->apply(in, out);
        out.download(frame_out);
    }
    else
        bg_subtractor->apply(frame_in, frame_out);
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

void drawROIs(cv::Mat& frame, std::vector<tk::dnn::box>& detected) {
    tk::dnn::box b;
    int x0, w, x1, y0, h, y1;
    int baseline = 0;
    float font_scale = 0.5;
    int thickness = 2;   

    // draw ROIs
    for(int i=0; i< detected.size(); i++) { 
        b           = detected[i];
        x0   		= b.x;
        x1   		= b.x + b.w;
        y0   		= b.y;
        y1   		= b.y + b.h;

        // draw rectangle
        cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0,0,0), cv::FILLED); 
    }
}

void drawBatchesDetOnOriginalFrame(cv::Mat & frame, const std::vector<std::vector<tk::dnn::box>>& batchDetected, const std::vector<cv::Rect>& or_rects, std::vector<std::string>& classesNames, cv::Scalar* colors){
    tk::dnn::box b;
    int x0, w, x1, y0, h, y1;
    int objClass;
    std::string det_class;

    int baseline = 0;
    float font_scale = 0.5;
    int thickness = 2;   

    for(int bi=0; bi<batchDetected.size(); ++bi){
        // draw dets
        for(int i=0; i<batchDetected[bi].size(); i++) { 
            b           = batchDetected[bi][i];
            b.x         = b.x+or_rects[bi].x;
            b.y         = b.y+or_rects[bi].y;
            x0   		= b.x;
            x1   		= b.x + b.w;
            y0   		= b.y;
            y1   		= b.y + b.h;
            det_class 	= classesNames[b.cl];

            // draw rectangle
            cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), colors[b.cl], 2); 

            // draw label
            cv::Size text_size = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
            cv::rectangle(frame, cv::Point(x0, y0), cv::Point((x0 + text_size.width - 2), (y0 - text_size.height - 2)), colors[b.cl], -1);                      
            cv::putText(frame, det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
        }
    }
}


int main(int argc, char *argv[]) {

    
    //calibration matrix needed for undistort
    std::string camera_id = "20936";
    cv::Mat calib_mat, dist_coeff;
    int o_width, o_height;
    readCalibrationMatrix("../data/calib_cameras/"+camera_id+".params", calib_mat, dist_coeff, o_width, o_height);

    //background subtractor
    bool on_gpu = true;
    cv::Ptr<cv::BackgroundSubtractor> bg_subtractor;
    if (on_gpu)
        bg_subtractor = cv::cuda::createBackgroundSubtractorMOG2();
    else
        bg_subtractor = cv::createBackgroundSubtractorKNN();

    //video capture
    bool gRun = true;
    cv::VideoCapture cap("../data/"+camera_id+".mp4", cv::CAP_FFMPEG);
    if(!cap.isOpened())
        gRun = false; 

    //roi batches
    std::vector<cv::Mat> batches;
    std::vector<cv::Rect> or_rects;
    bool use_batches = true;

    //network
    std::string net = "yolo3_512_fp32.rt";
    if(!use_batches) net = "yolo3_berkeley_fp32.rt";
    char ntype = 'y';
    int n_classes = 80;
    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;  
    tk::dnn::DetectionNN *detNN;  
    
    switch(ntype){
        case 'y':
            detNN = &yolo;
            break;
        case 'c':
            detNN = &cnet;
            break;
        case 'm':
            detNN = &mbnet;
            n_classes++;
            break;
        default:
        FatalError("Network type not allowed (3rd parameter)\n");
    }
    detNN->init(net, n_classes, MAX_BATCHES);

    //visualization
    bool show = true;

    //resize and undistort
    const int new_width  = 960;
    const int new_height = 540;
    cv::Mat resized_frame, map1, map2, undistort;
    bool first_iteration = true;
    uint8_t *d_input, *d_output; 
    float *d_map1, *d_map2;

    //frame capture
    cv::Mat frame;
    std::vector<cv::Mat> batch_frame, batch_frame2;
    std::vector<cv::Mat> batch_dnn_input;

    //roi
    cv::Mat roi(cv::Size(new_width, new_height), CV_8UC1);
    roi.setTo(cv::Scalar(255,255,255));

    //disparity
    cv::Mat canny, pre_canny, canny_RGB, pre_canny_RGB, disparity_canny;
    cv::Mat old_frame, disparity_optical_flow;
    cv::Mat bg_suppressed, fg_boxes;
    
    //draw
    cv::Scalar colors[256];
    for(int c=0; c<n_classes; ++c) {
        int offset = c*123457 % n_classes;
        float r = getColor(2, offset, n_classes);
        float g = getColor(1, offset, n_classes);
        float b = getColor(0, offset, n_classes);
        colors[c] = cv::Scalar(int(255.0*b), int(255.0*g), int(255.0*r));
    }
    
    //profiler
    edge::Profiler prof("rois");
    cv::Mat backsupp;
    
    edge::BackGroundSuppression1 bs1(algo, new_width, new_height);
    edge::BackGroundSuppression2 bs2(algo, new_width, new_height, n_classes);

    /* mode -----
    'n' --> the normal detection
    'b' --> use batches 
    'c' --> collage mode
    'd' --> dark mode: set the background with black color
    */
    char mode = 'b';
    if(use_batches && mode != 'b' || !use_batches && mode=='b') {
        std::cout<<"ERROR: check use_batches and the mode\n";
    }
    while(gRun) {
        prof.tick("total time");

        //frame reading
        prof.tick("get frame");
        cap >> frame; 
        if(!frame.data) break;
        prof.tock("get frame");

        //resize
        prof.tick("resize");
        cv::resize (frame, resized_frame, cv::Size(new_width, new_height)); 
        prof.tock("resize");

        //undistort        
        prof.tick("undistort");
        if (first_iteration){
            calib_mat.at<double>(0,0)*=  double(new_width) / double(o_width);
            calib_mat.at<double>(0,2)*=  double(new_width) / double(o_width);
            calib_mat.at<double>(1,1)*=  double(new_width) / double(o_width);
            calib_mat.at<double>(1,2)*=  double(new_width) / double(o_width);
            
            cv::initUndistortRectifyMap(calib_mat, dist_coeff, cv::Mat(), calib_mat, resized_frame.size(), CV_32F, map1, map2);
            
            checkCuda( cudaMalloc(&d_input, resized_frame.cols*resized_frame.rows*resized_frame.channels()*sizeof(uint8_t)) );
            checkCuda( cudaMalloc(&d_output, resized_frame.cols*resized_frame.rows*resized_frame.channels()*sizeof(uint8_t)) );
            checkCuda( cudaMalloc(&d_map1, map1.cols*map1.rows*map1.channels()*sizeof(float)) );
            checkCuda( cudaMalloc(&d_map2, map2.cols*map2.rows*map2.channels()*sizeof(float)) );

            checkCuda( cudaMemcpy(d_map1, (float*)map1.data,  map1.cols*map1.rows*map1.channels()*sizeof(float), cudaMemcpyHostToDevice));
            checkCuda( cudaMemcpy(d_map2, (float*)map2.data,  map2.cols*map2.rows*map2.channels()*sizeof(float), cudaMemcpyHostToDevice));

            undistort = resized_frame.clone();
        }
        checkCuda( cudaMemcpy(d_input, (uint8_t*)resized_frame.data,  resized_frame.cols*resized_frame.rows*resized_frame.channels()*sizeof(uint8_t), cudaMemcpyHostToDevice));
        remap(d_input, new_width, new_height, 3, d_map1, d_map2, d_output, new_width , new_height, 3);
        checkCuda( cudaMemcpy((uint8_t*)undistort.data , d_output, resized_frame.cols*resized_frame.rows*resized_frame.channels()*sizeof(uint8_t), cudaMemcpyDeviceToHost));

        batch_frame.clear();
        batch_frame.push_back(undistort.clone());
        prof.tock("undistort");
        
        //get disparity frame
        // prof.tick("disparity");
        // disparity_canny = getDisparityCanny(undistort,first_iteration, canny, pre_canny, canny_RGB, pre_canny_RGB);
        // disparity_optical_flow = getDisparityOpticalFlow(undistort, old_frame, first_iteration);
        // prof.tock("disparity");

        prof.tick("det process");
        switch (mode)
        {
        case 'b':
            //background suppression
            prof.tick("background");
            suppressBackground(undistort, bg_suppressed, bg_subtractor,on_gpu);
            prof.tock("background");

            //boxes extraction
            prof.tick("extract boxes");
            batches.clear();
            or_rects.clear();
            getBatchesFromMovingObjs(undistort, bg_suppressed, fg_boxes, batches, or_rects);
            prof.tock("extract boxes");

            //inference
            prof.tick("inference");
            batch_dnn_input.clear();
            for(auto b:batches)
                batch_dnn_input.push_back(b.clone());
            detNN->update(batch_dnn_input, batch_dnn_input.size());
            prof.tock("inference");
            
            //draw boxes
            prof.tick("draw");
            drawBatchesDetOnOriginalFrame(batch_frame[0], detNN->batchDetected, or_rects, detNN->classesNames, colors);
            prof.tock("draw");
            break;
        case 'n':
            //inference
            prof.tick("inference");
            batch_dnn_input.clear();
            batch_dnn_input.push_back(undistort.clone());
            detNN->update(batch_dnn_input, batch_dnn_input.size());
            prof.tock("inference");
            
            //draw boxes
            prof.tick("draw");
            detNN->draw(batch_frame);
            prof.tock("draw");
            break;
        case 'c':
            prof.tick("backgroundsuppression2");
            backsupp = bs2.update(undistort);
            prof.tock("backgroundsuppression2");
            prof.tick("inference");
            //inference
            batch_frame2.clear();
            batch_frame2.push_back(backsupp.clone());
            batch_dnn_input.clear();
            batch_dnn_input.push_back(backsupp.clone());
            detNN->update(batch_dnn_input, batch_dnn_input.size());
            prof.tock("inference");
            prof.tick("draw");
            detNN->draw(batch_frame2);
            prof.tock("draw");
            prof.tick("draw-2");
            bs2.drawTiles(batch_frame, detNN->batchDetected, detNN->classesNames);
            prof.tock("draw-2");
            break;
        case 'd':
            prof.tick("backgroundsuppression1");
            backsupp = bs1.update(undistort);
            prof.tock("backgroundsuppression1");
            prof.tick("inference");
            //inference
            batch_frame2.clear();
            batch_frame2.push_back(backsupp.clone());
            batch_dnn_input.clear();
            batch_dnn_input.push_back(backsupp.clone());
            detNN->update(batch_dnn_input, batch_dnn_input.size());
            prof.tock("inference");
            prof.tick("draw");
            detNN->draw(batch_frame2);
            detNN->draw(batch_frame);
            prof.tock("draw");
            break;
        default:
            break;
        }
        prof.tock("det process");

        //visualization
        prof.tick("show");
        if(show){

            // //visualize each batch 
            // detNN->draw(batches);
            // for(auto& b:batches){
            //     if(b.rows*b.cols > 3000){
            //     cv::imshow("batch", b);
            //     cv::waitKey(0);
            //     }
            // }

            // //visualize canny or optical flow
            // cv::imshow("disparity Canny ", disparity_canny);
            // cv::imshow("disparity Optical Flow ", disparity_optical_flow);

            // // visualize black ROIs on white frane
            // drawROIs(roi, detNN->detected);
            // cv::imshow("roi", roi);

            // // viosualize baground suppression and extracted boxes on frame
            // cv::imshow("Background Suppression", bg_suppressed);
            // cv::imshow("boxes of foreground objects", fg_boxes);
            switch (mode)
            {
            case 'c':
            case 'd':
                cv::imshow("backsupp", backsupp);
                cv::imshow("backsupp det", batch_frame2[0]);
            default:
                break;
            }
            cv::imshow("detection", batch_frame[0]);

            cv::waitKey(1);
        }
        prof.tock("show");

        if(first_iteration)
            first_iteration = false;

        prof.tock("total time");
        prof.printStats(500);
    }

    // imwrite( "roi_"+camera_id+".jpg", roi);
    return 0;
}

