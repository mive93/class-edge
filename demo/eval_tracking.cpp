#include <iostream>
#include "tkDNN/CenternetDetection.h"
#include "tkDNN/MobilenetDetection.h"
#include "tkDNN/Yolo3Detection.h"
#include "tkDNN/BoundingBox.h"
#include "configuration.h"
#include "Profiler.h"
#include "camera_elaboration.h"
#include "roi.h"
#include <opencv2/imgproc.hpp>

std::vector<tk::dnn::box> readFrameGroundtruth(std::ifstream& gt, std::streampos& oldpos,const int frame_id){
    std::string line;
    std::vector<tk::dnn::box> groundtruth;
    oldpos = gt.tellg();
    while(getline(gt,line)){
        std::stringstream linestream(line);
        std::string value;
        std::vector<int> values;

        while(getline(linestream,value,','))
            values.push_back(std::stoi(value));
        if(values[0]!= frame_id){
            gt.seekg (oldpos);
            break;
        }
        oldpos = gt.tellg();

        tk::dnn::box b;
        b.x         = values[2];
        b.y         = values[3];
        b.w         = values[4];
        b.h         = values[5];
        b.cl        = values[6];
        if(b.cl == CAR_ID_AICC)       b.cl = CAR_ID; 
        else if(b.cl == TRUCK_ID_AICC)  b.cl = TRUCK_ID; 
        b.prob      = 1;
        
        groundtruth.push_back(b);
    } 
    // std::cout<<groundtruth.size()<<std::endl;
    return groundtruth;
}

int main(int argc, char **argv)
{

    //mode
    edge::DetProcess_t mode = edge::OF_BATCHES;
    if(argc > 1 && atoi(argv[1]) <= int(edge::COLLAGE))
        mode = edge::DetProcess_t(atoi(argv[1])); 

    std::vector<int> gt_ids = {7,9,10};
    // std::vector<int> gt_ids = {7,8,33,27};
    // std::vector<int> gt_ids = {6,7,8,9,10,16,17,18,19,20,21,22,23,24,25,26,27,28,29,33,34,35,36};

    if(!fileExist("../data/gt/c007/vdo.avi")){
        system("wget https://cloud.hipert.unimore.it/s/g3f77BeoziMxPdB/download -O ../data/gt.zip");
        system("unzip ../data/gt.zip -d ../data/");
        system("rm ../data/gt.zip");
    }

    //detection
    std::vector<tk::dnn::box> detected;
    cv::Mat frame;
    std::vector<cv::Mat> batch_dnn_input;

    //network
    tk::dnn::DetectionNN *detNN;  
    int n_classes = 80;
    if(mode == edge::BS_BATCHES || mode == edge::CANNY_BATCHES || mode == edge::OF_BATCHES) {
        std::string net = "yolo3_64_fp32.rt";
        detNN = new tk::dnn::Yolo3Detection();
        detNN->init(net, n_classes, MAX_BATCHES);
    }
    else{
        n_classes = 10;
        std::string net = "yolo3_berkeley_fp32.rt";
        detNN = new tk::dnn::Yolo3Detection();
        detNN->init(net, n_classes);
    }

    //background subtractor
    bool on_gpu = true;
    edge::BackGroundSuppression bs(on_gpu);
    int width = 1920;
    int height = 1080;
    edge::BackGroundSuppression1 bs1(on_gpu, width, height);
    edge::BackGroundSuppression2 bs2(on_gpu, width, height, n_classes);

    //roi 
    cv::Mat old_frame, pre_canny;

    //profiler
    edge::Profiler prof("tracker eval");

    //visualization
    show = false;
    bool show_cam = true;
    if(show){
        viewer = new edge::EdgeViewer(1);
        viewer->setWindowName("tracking");
        viewer->setBackground(tk::gui::color::DARK_GRAY);
        viewer->initOnThread(false);
        viewer->setClassesNames(detNN->classesNames);
        viewer->setColors(detNN->classes);
        viewer->bindCamera(0, &show_cam);
    }   

    //outputs
    std::string mkdir_cmd = "mkdir ../data/dets_"+std::to_string(int(mode));
    system(mkdir_cmd.c_str());
    system("mkdir res");
    std::ofstream out_times("res/times_"+std::to_string(int(mode))+".csv");

    for(auto& id:gt_ids){
        //convert id to 00id
        std::string cam_id = std::to_string(id);
        std::string cam_id_filled = std::string(3 - cam_id.length(), '0') + cam_id;

        //read projection matrix
        cv::Mat prjMat, invPrjMat;
        readProjectionMatrix("../data/gt/c"+cam_id_filled+"/calibration.txt", invPrjMat);
        prjMat = invPrjMat.inv();

        //geodetic converter
        tk::common::GeodeticConverter geoConv;
        if(id < 10)
            geoConv.initialiseReference(42.491916, -90.723723, 0);
        else
            geoConv.initialiseReference(42.498780, -90.686393, 0);

        //video capture
        gRun = true;
        cv::VideoCapture cap("../data/gt/c"+cam_id_filled+"/vdo.avi");
        if(!cap.isOpened()){
            gRun = false; 
            FatalError("Can't open the video stream");
        }

        //tracker
        float   dt              = 0.03;
        int     n_states        = 5;
        int     initial_age     = 5;
        bool    tr_verbose      = false;
        tracking::Tracking t(n_states, dt, initial_age);
        std::vector<tracking::obj_m>    cur_frame;
        double east, north;  

        //groundtruth
        std::string gt_filename = "../data/gt/c"+cam_id_filled+"/mtsc/mtsc_deepsort_yolo3.txt";
        std::ifstream gt(gt_filename);
        std::streampos oldpos;
        
        //output
        std::ofstream out_f("../data/dets_"+std::to_string(int(mode))+"/c"+cam_id_filled+".txt");
        

        int frame_id = 1;
        bool first_iteration = true;
        while(gRun) {
            prof.tick("total time");

            //frame reading
            prof.tick("get frame");
            cap >> frame; 
            if(!frame.data) break;
            prof.tock("get frame");

            //detection process
            prof.tick("det process");
            detected.clear();
            detected = detectionProcess(mode,detNN, frame, batch_dnn_input, prof, &bs, first_iteration, &pre_canny, &old_frame, &bs1, &bs2);
            prof.tock("det process");

            //feed the tracker
            prof.tick("Tracker feeding");
            cur_frame.clear();
            for(auto d:detected){
                if(d.cl == CAR_ID){
                    convertCameraPixelsToMapMeters(d.x + d.w / 2, d.y + d.h, d.cl, prjMat, north, east, geoConv);
                    tracking::obj_m obj;
                    obj.frame   = 0;
                    obj.cl      = d.cl;
                    obj.x       = north;
                    obj.y       = east;
                    obj.w       = d.w; 
                    obj.h       = d.h;
                    cur_frame.push_back(obj);
                }
            }
            t.track(cur_frame,tr_verbose);
            prof.tock("Tracker feeding");
            

            //feed the viewer
            prof.tick("Viewer feeding");
            std::vector<tk::dnn::box>  tr_detected;
            auto tr_lines = getTrackingLines(t, invPrjMat, geoConv, tr_detected, nullptr, 0,1, 1, false, &out_f, frame_id);

            if(show){
                std::vector<edge::tracker_line> lines;
                viewer->setFrameData(frame, tr_detected, lines, 0, readFrameGroundtruth(gt, oldpos, frame_id));
            }
            prof.tock("Viewer feeding");

            prof.tock("total time");
            prof.printStats(500, &out_times);

            frame_id++;
            if(first_iteration) first_iteration = false;
        }

        out_f.close();
        
    }
    
    if(show){
        viewer->close();
        viewer->joinThread();
    }

    out_times.close();

    
    std::string python_cmd = "python3 -m motmetrics.apps.eval_motchallenge ../data/gt_mtsc_yolo3_deepsort/ ../data/dets_"+std::to_string(int(mode))+"/ > res/metrics_"+std::to_string(int(mode))+".txt";
    // system(python_cmd.c_str());

    return EXIT_SUCCESS;
}
