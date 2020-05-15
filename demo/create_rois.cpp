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
    return disparity;
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


int main(int argc, char *argv[]) {

    
    //calibration matrix needed for undistort
    std::string camera_id = "20936";
    cv::Mat calib_mat, dist_coeff;
    int o_width, o_height;
    readCalibrationMatrix("../data/calib_cameras/"+camera_id+".params", calib_mat, dist_coeff, o_width, o_height);

    //video capture
    bool gRun = true;
    cv::VideoCapture cap("../data/"+camera_id+".mp4", cv::CAP_FFMPEG);
    if(!cap.isOpened())
        gRun = false; 

    //network
    std::string net = "yolo3_berkeley_fp32.rt";
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
    detNN->init(net, n_classes);

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
    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    //roi
    cv::Mat roi(cv::Size(new_width, new_height), CV_8UC1);
    roi.setTo(cv::Scalar(255,255,255));

    //disparity
    cv::Mat canny, pre_canny, canny_RGB, pre_canny_RGB, disparityCanny, old_frame, disparityOpticalFlow;

    while(gRun) {
        
        //frame reading
        cap >> frame; 
        if(!frame.data) break;
        
        //resize
        cv::resize (frame, resized_frame, cv::Size(new_width, new_height)); 

        //undistort        
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
        batch_frame.push_back(undistort);

        //get disparity frame
        // disparityCanny = getDisparityCanny(undistort,first_iteration, canny, pre_canny, canny_RGB, pre_canny_RGB);
        disparityOpticalFlow = getDisparityOpticalFlow(undistort, old_frame, first_iteration);
        
        //inference
        batch_dnn_input.clear();
        batch_dnn_input.push_back(undistort.clone());
        detNN->update(batch_dnn_input);
        detNN->draw(batch_frame);

        //visualization
        if(show){
            cv::imshow("detection", batch_frame[0]);
            // cv::imshow("disparity Canny ", disparityCanny);
            cv::imshow("disparity Optical Flow ", disparityOpticalFlow);

            drawROIs(roi, detNN->detected);
            cv::imshow("roi", roi);

            cv::waitKey(1);
        }

        if(first_iteration)
            first_iteration = false;
    }

    imwrite( "roi_"+camera_id+".jpg", roi);
    return 0;
}

