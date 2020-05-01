#include "camera_elaboration.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "tkDNN/utils.h"

#include "Profiler.h"
#include <opencv2/cudawarping.hpp>


void pixel2GPS(const int x, const int y, double &lat, double &lon)
{
    //conversion from pixels to GPS, via georeferenced map parameters
    double xoff, a, b, yoff, d, e;
    xoff    = adfGeoTransform[0];
    a       = adfGeoTransform[1];
    b       = adfGeoTransform[2];
    yoff    = adfGeoTransform[3];
    d       = adfGeoTransform[4];
    e       = adfGeoTransform[5];

    lon     = a * x + b * y + xoff;
    lat     = d * x + e * y + yoff;
}
void GPS2pixel(double lat, double lon, int &x, int &y)
{
    //conversion from GPS to pixels, via georeferenced map parameters
    x = int(round( (lon - adfGeoTransform[0]) / adfGeoTransform[1]) );
    y = int(round( (lat - adfGeoTransform[3]) / adfGeoTransform[5]) );
}

void convertCameraPixelsToMapMeters(const int x, const int y, const int cl, const cv::Mat& prj_mat, double& north, double& east)
{
    double latitude, longitude;
    double up;
    
    //transform camera pixel into georeferenced map pixel
    std::vector<cv::Point2f> x_y, ll;
    x_y.push_back(cv::Point2f(x, y));
    cv::perspectiveTransform(x_y, ll, prj_mat);

    //tranform to map pixel into GPS
    pixel2GPS(ll[0].x, ll[0].y, latitude, longitude);

    //conversion from GPS to meters 
    geoConv.geodetic2Enu(latitude, longitude, 0, &north, &east, &up);    
}

std::vector<edge::tracker_line> getTrackingLines(const tracking::Tracking& t, const cv::Mat& inv_prj_mat, const int cam_id, const float scale_x, const float scale_y, bool verbose){
    std::vector<edge::tracker_line>  lines;
    std::vector<cv::Point2f> map_pixels;
    std::vector<cv::Point2f> camera_pixels;

    double north, east, up;
    double latitude, longitude, altitude;
    int map_pix_x, map_pix_y; 

    for(auto tr: t.trackers){
        if(tr.predList.size()){
            edge::tracker_line line;

            map_pixels.clear();
            camera_pixels.clear();
            for(int i=0; i < tr.predList.size(); ++i){
                //convert from meters to GPS
                geoConv.enu2Geodetic(tr.predList[i].x, tr.predList[i].y, 0, &latitude, &longitude, &altitude);
                //convert from GPS to map pixels
                GPS2pixel(latitude, longitude, map_pix_x, map_pix_y);
                map_pixels.push_back(cv::Point2f(map_pix_x, map_pix_y));
            }

            //transform map pixels to camers pixels
            cv::perspectiveTransform(map_pixels, camera_pixels, inv_prj_mat);
            
            //convert into viewer coordinates
            for(auto cp: camera_pixels)
            {
                if(verbose)
                    std::cout<<"x:\t"<<cp.x<<"\t y:\t"<<cp.y<<std::endl;
                line.points.push_back(viewer->convertPosition(cp.x*scale_x, cp.y*scale_y, -0.004, cam_id));
            }
            line.color = tk::gui::Color_t {tr.r, tr.g, tr.b, 255};
            lines.push_back(line);
        }
    }
    return lines;
}

void prepareMessage(const tracking::Tracking& t, MasaMessage& message)
{
    message.objects.clear();
    double latitude, longitude, altitude;
    int i = 0;
    for(auto tr: t.trackers){
        if(tr.predList.size()){
            //convert from meters to GPS
            i = tr.predList.size() -1;
            geoConv.enu2Geodetic(tr.predList[i].x, tr.predList[i].y, 0, &latitude, &longitude, &altitude);
            //add RoadUser to the message
            message.objects.push_back(getRoadUser(latitude, longitude, tr.predList[i].vel, tr.predList[i].yaw, tr.cl));
        }
    }

    message.t_stamp_ms = getTimeMs();
    message.num_objects = message.objects.size();
}

void *elaborateSingleCamera(void *ptr)
{
    edge::camera* cam = (edge::camera*) ptr;
    std::cout<<"Starting camera: "<< cam->id << std::endl;

    pthread_t video_cap;
    edge::video_cap_data data;
    data.input = (char*)cam->input.c_str();

    if(cam->show)
        viewer->bindCamera(cam->id);
    
    if (pthread_create(&video_cap, NULL, readVideoCapture, (void *)&data)){
        fprintf(stderr, "Error creating thread\n");
        return (void *)1;
    }
    
    //instantiate the communicator and the socket
    Communicator<MasaMessage> communicator;
    communicator.open_client_socket((char*)cam->ipCommunicator.c_str(), cam->portCommunicator);
    int socket = communicator.get_socket();
    MasaMessage message;
    
    //initiate the tracker
    float   dt              = 0.03;
    int     n_states        = 5;
    int     initial_age     = 5;
    bool    tr_verbose      = false;
    tracking::Tracking t(n_states, dt, initial_age);
    

    cv::Mat dnn_input;
    cv::Mat distort;
    cv::Mat map1, map2;

    std::vector<tk::dnn::box>       detected;
    std::vector<tracking::obj_m>    cur_frame;

    double north, east;
    bool verbose = false;
    bool first_iteration = true; 

    cv::cuda::GpuMat frame_gpu, map1_gpu, map2_gpu;
    float o_width, o_height;
    float scale_x = 1;
    float scale_y = 1;

    cam->show = true;

    //profiling
    edge::Profiler prof(std::to_string(cam->id));

    while(gRun){
        prof.tick("Total time");
        if(data.frame.data) {
            //copy the frame that the last frame read by the video capture thread
            prof.tick("Copy frame");
            data.mtxF.lock();
            distort     = data.frame.clone();
            o_width     = data.oWidth;
            o_height    = data.oHeight;
            data.mtxF.unlock();
            scale_x     = o_width / distort.cols;
            scale_y     = o_height / distort.rows;
            prof.tock("Copy frame");
                       
            // undistort 
            prof.tick("Undistort");
            if (first_iteration){
                cam->calibMat *=  distort.cols / o_width ;
                cv::initUndistortRectifyMap(cam->calibMat, cam->distCoeff, cv::Mat(), cam->calibMat, cv::Size(o_width, o_height), CV_32F, map1, map2);
                map1_gpu.upload(map1);
                map2_gpu.upload(map2);
                first_iteration = false;
            }
            cv::Mat frame;
            cv::cuda::GpuMat distort_gpu;
            distort_gpu.upload(distort);
            cv::cuda::remap(distort_gpu, frame_gpu, map1_gpu, map2_gpu, cv::INTER_LINEAR);
            frame_gpu.download(frame);
            // cv::remap(distort, frame, map1, map2, cv::INTER_LINEAR);
            prof.tock("Undistort");

            //inference
            prof.tick("Inference");
            dnn_input = frame.clone();  
            cam->detNN->update(dnn_input);
            detected= cam->detNN->detected;
            prof.tock("Inference");

            //feed the tracker
            prof.tick("Tracker feeding");
            cur_frame.clear();
            for(auto d:detected){
                if(d.cl < 6){
                    convertCameraPixelsToMapMeters((d.x + d.w / 2)*scale_x, (d.y + d.h)*scale_y, d.cl, cam->prjMat, north, east);
                    tracking::obj_m obj;
                    obj.frame   = 0;
                    obj.cl      = d.cl;
                    obj.x       = north;
                    obj.y       = east;
                    cur_frame.push_back(obj);
                }
            }
            t.track(cur_frame,tr_verbose);
            prof.tock("Tracker feeding");

            //feed the viewer
            prof.tick("Viewer feeding");
            if(cam->show)
                viewer->setFrameData(frame, detected, getTrackingLines(t, cam->invPrjMat, cam->id, 1/scale_x, 1/scale_y,verbose), cam->id);
            prof.tock("Viewer feeding");

            prof.tick("Prepare message");
            //send the data if the message is not empty
            prepareMessage(t, message);
            if (!message.objects.empty()){
                communicator.send_message(&message, cam->portCommunicator);
                // std::cout<<"message sent!"<<std::endl;
            }
            prof.tock("Prepare message");
        }
        prof.tock("Total time");    
        prof.printStats();
    }
    return (void *)0;
}