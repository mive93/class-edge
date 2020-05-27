#include <iostream>
#include "tkDNN/CenternetDetection.h"
#include "tkDNN/MobilenetDetection.h"
#include "tkDNN/Yolo3Detection.h"
#include "tkDNN/BoundingBox.h"
#include "configuration.h"
#include "Profiler.h"
#include "camera_elaboration.h"
#include <opencv2/imgproc.hpp>

std::vector<tk::dnn::box> readFrameGroundtruth(std::ifstream& gt, std::streampos& oldpos,const int frame_id){
    std::string line;
    std::vector<tk::dnn::box> groundtruth;
    groundtruth.clear();
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
        b.prob      = 1;
        
        groundtruth.push_back(b);
    } 
    // std::cout<<groundtruth.size()<<std::endl;
    return groundtruth;
}

int main(int argc, char **argv)
{

    if(!fileExist("../data/gt/c007/vdo.avi")){
        system("wget https://cloud.hipert.unimore.it/s/g3f77BeoziMxPdB/download -O ../data/gt.zip");
        system("unzip ../data/gt.zip -d ../data/");
        system("rm ../data/gt.zip");
    }

    std::vector<edge::camera> cameras = configure(argc, argv, true);

    //detection
    std::vector<tk::dnn::box> detected;
    cv::Mat frame, dnn_input;
    
    //profiler
    edge::Profiler prof("tracker eval");

    //visualization
    bool show = true;
    if(show){
        viewer = new edge::EdgeViewer(cameras.size());
        viewer->setWindowName("tracking");
        viewer->setBackground(tk::gui::color::DARK_GRAY);
        viewer->initOnThread(false);
        viewer->setClassesNames(cameras[0].detNN->classesNames);
        viewer->setColors(cameras[0].detNN->classes);
        //binding camera to viewer
        for(auto& camera:cameras){
            camera.show = false;
            viewer->bindCamera(camera.id, &camera.show);

        }
    }   

    system("mkdir ../data/dets");

    for(auto& camera:cameras){
        if(show) camera.show = true;

        //video capture
        gRun = true;
        cv::VideoCapture cap(camera.input);
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
        std::string cam_id = std::to_string(camera.id);
        std::string cam_id_filled = std::string(3 - cam_id.length(), '0') + cam_id;
        std::string gt_filename = "../data/gt/c"+cam_id_filled+"/gt/gt.txt";
        std::ifstream gt(gt_filename);
        std::streampos oldpos;
        
        //output
        std::ofstream out_f("../data/dets/c"+cam_id_filled+".txt");

        int frame_id = 1;
        while(gRun) {
            prof.tick("total time");

            //frame reading
            prof.tick("get frame");
            cap >> frame; 
            if(!frame.data) break;
            prof.tock("get frame");

            //inference
            prof.tick("inference");
            dnn_input = frame.clone();
            camera.detNN->update(dnn_input);
            detected= camera.detNN->detected;
            prof.tock("inference");

            //feed the tracker
            prof.tick("Tracker feeding");
            cur_frame.clear();
            for(auto d:detected){
                if(d.cl == 1){
                    convertCameraPixelsToMapMeters(d.x + d.w / 2, d.y + d.h, d.cl, camera.prjMat, north, east, camera.geoConv);
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
            auto tr_lines = getTrackingLines(t, camera.invPrjMat, camera.geoConv, tr_detected, nullptr, camera.id,1, 1, false, &out_f, frame_id);

            if(show && camera.show){
                std::vector<edge::tracker_line> lines;
                viewer->setFrameData(frame, tr_detected, tr_lines, camera.id, readFrameGroundtruth(gt, oldpos, frame_id));
            }
            prof.tock("Viewer feeding");

            prof.tock("total time");
            // prof.printStats(100);

            frame_id++;
        }

        out_f.close();
        camera.show = false;
    }
    
    if(show) viewer->joinThread();

    system("python3 -m motmetrics.apps.eval_motchallenge ../data/gt/ ../data/dets/");

    return EXIT_SUCCESS;
}
