#include "roi.h"

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

    //mode
    edge::DetProcess_t mode = edge::OF_BATCHES;
    if(argc > 1 && atoi(argv[1]) < int(edge::COLLAGE))
        mode = edge::DetProcess_t(atoi(argv[1])); 

    //calibration matrix needed for undistort
    std::string camera_id = "20936";
    cv::Mat calib_mat, dist_coeff;
    int o_width, o_height;
    readCalibrationMatrix("../data/calib_cameras/"+camera_id+".params", calib_mat, dist_coeff, o_width, o_height);

    //resize and undistort
    const int new_width  = 960;
    const int new_height = 540;
    cv::Mat resized_frame, map1, map2, undistort;
    bool first_iteration = true;
    uint8_t *d_input, *d_output; 
    float *d_map1, *d_map2;

    //video capture
    bool gRun = true;
    cv::VideoCapture cap("../data/"+camera_id+".mp4", cv::CAP_FFMPEG);
    if(!cap.isOpened())
        gRun = false; 

    //network
    std::string net = "yolo3_64_fp32.rt";
    char ntype = 'y';
    int n_classes = 80;
    if(mode == edge::FULL_IMG || mode == edge::FULL_IMG_BS || mode == edge::COLLAGE) {
        net = "yolo3_berkeley_fp32.rt";
        n_classes = 10;
    }
    
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
    std::vector<tk::dnn::box> detected;    

    //frame capture
    cv::Mat frame;
    std::vector<cv::Mat> batch_frame,batch_dnn_input;

    //background subtractor
    bool on_gpu = true;
    edge::BackGroundSuppression bs(on_gpu);
    edge::BackGroundSuppression1 bs1(on_gpu, new_width, new_height);
    edge::BackGroundSuppression2 bs2(on_gpu, new_width, new_height, n_classes);

    //roi
    cv::Mat roi(cv::Size(new_width, new_height), CV_8UC1);
    roi.setTo(cv::Scalar(255,255,255));
    cv::Mat old_frame, pre_canny;

    //visualization
    bool show = true;
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
        
        prof.tick("det process");
        detected.clear();
        detected = detectionProcess(mode,detNN, undistort, batch_dnn_input, prof, &bs, first_iteration, &pre_canny, &old_frame, &bs1, &bs2);
        prof.tock("det process");

        //draw boxes
        prof.tick("draw");
        if(detNN->batchDetected.size()){
            detNN->batchDetected[0] = detected;
            detNN->draw(batch_frame);
        }
        prof.tock("draw");

        //visualization
        prof.tick("show");
        if(show){
            cv::imshow("detection", batch_frame[0]);
            cv::waitKey(1);
        }
        prof.tock("show");

        if(first_iteration)
            first_iteration = false;

        prof.tock("total time");
        prof.printStats(500);
    }
    
    return 0;
}

