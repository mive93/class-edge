#include "camera_elaboration.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "tkDNN/utils.h"

#include "Profiler.h"

#include "undistort.h"

void pixel2GPS(const int x, const int y, double &lat, double &lon, double* adfGeoTransform)
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
void GPS2pixel(double lat, double lon, int &x, int &y, double* adfGeoTransform)
{
    //conversion from GPS to pixels, via georeferenced map parameters
    x = int(round( (lon - adfGeoTransform[0]) / adfGeoTransform[1]) );
    y = int(round( (lat - adfGeoTransform[3]) / adfGeoTransform[5]) );
}

void convertCameraPixelsToMapMeters(const int x, const int y, const int cl, edge::camera& cam, double& north, double& east)
{
    double latitude, longitude;
    double up;
    
    //transform camera pixel into georeferenced map pixel
    std::vector<cv::Point2f> x_y, ll;
    x_y.push_back(cv::Point2f(x, y));
    cv::perspectiveTransform(x_y, ll, cam.prjMat);

    //tranform to map pixel into GPS
    pixel2GPS(ll[0].x, ll[0].y, latitude, longitude, cam.adfGeoTransform);

    //conversion from GPS to meters 
    cam.geoConv.geodetic2Enu(latitude, longitude, 0, &north, &east, &up);    
}

std::vector<edge::tracker_line> getTrackingLines(const tracking::Tracking& t, edge::camera& cam, const float scale_x, const float scale_y, bool verbose){
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
                cam.geoConv.enu2Geodetic(tr.predList[i].x, tr.predList[i].y, 0, &latitude, &longitude, &altitude);
                //convert from GPS to map pixels
                GPS2pixel(latitude, longitude, map_pix_x, map_pix_y, cam.adfGeoTransform);
                map_pixels.push_back(cv::Point2f(map_pix_x, map_pix_y));
            }

            //transform map pixels to camers pixels
            cv::perspectiveTransform(map_pixels, camera_pixels, cam.invPrjMat);
            
            //convert into viewer coordinates
            for(auto cp: camera_pixels)
            {
                if(verbose)
                    std::cout<<"x:\t"<<cp.x<<"\t y:\t"<<cp.y<<std::endl;
                line.points.push_back(viewer->convertPosition(cp.x*scale_x, cp.y*scale_y, -0.004, cam.id));
            }
            line.color = tk::gui::Color_t {tr.r, tr.g, tr.b, 255};
            lines.push_back(line);
        }
    }
    return lines;
}

void prepareMessage(const tracking::Tracking& t, MasaMessage& message, tk::common::GeodeticConverter& geoConv, const int cam_id, edge::Dataset_t dataset)
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
            if(checkClass(tr.cl, dataset))
                message.objects.push_back(getRoadUser(latitude, longitude, tr.predList[i].vel, tr.predList[i].yaw, tr.cl, dataset));
        }
    }
    message.cam_idx = cam_id;
    message.t_stamp_ms = getTimeMs();
    message.num_objects = message.objects.size();
}

void *elaborateSingleCamera(void *ptr)
{
    edge::camera* cam = (edge::camera*) ptr;
    std::cout<<"Starting camera: "<< cam->id << std::endl;

    pthread_t video_cap;
    edge::video_cap_data data;
    data.input  = (char*)cam->input.c_str();
    data.width  = cam->streamWidth;
    data.height = cam->streamHeight;

    if(show)
        viewer->bindCamera(cam->id, &cam->show);
    
    if (pthread_create(&video_cap, NULL, readVideoCapture, (void *)&data)){
        fprintf(stderr, "Error creating thread\n");
        return (void *)1;
    }
    
    //instantiate the communicator and the socket
    // Communicator<MasaMessage> communicator;
    // communicator.open_client_socket((char*)cam->ipCommunicator.c_str(), cam->portCommunicator);
    // int socket = communicator.get_socket();
    // MasaMessage message;
    
    //initiate the tracker
    float   dt              = 0.03;
    int     n_states        = 5;
    int     initial_age     = 5;
    bool    tr_verbose      = false;
    tracking::Tracking t(n_states, dt, initial_age);
    

    cv::Mat frame;
    std::vector<cv::Mat> dnn_input;
    // cv::Mat distort;
    // cv::Mat map1, map2;

    std::vector<tk::dnn::box>       detected;
    // std::vector<tracking::obj_m>    cur_frame;

    // double north, east;
    // bool ce_verbose = false;
    // bool first_iteration = true; 

    // float scale_x   = cam->calibWidth  / cam->streamWidth;
    // float scale_y   = cam->calibHeight / cam->streamHeight;

    // uint8_t *d_input, *d_output; 
    // float *d_map1, *d_map2;

    //profiling
    edge::Profiler prof(std::to_string(cam->id));

    std::vector<edge::tracker_line> void_lines;

    while(gRun){
        prof.tick("Total time");
            
        //copy the frame that the last frame read by the video capture thread
        prof.tick("Copy frame");
        data.mtxF.lock();
        frame = data.frame.clone();
        data.mtxF.unlock();
        prof.tock("Copy frame");
        
        if(frame.data) {
            // undistort 
            // prof.tick("Undistort");
            // if (first_iteration){
            //     cam->calibMat.at<double>(0,0)*=  double(cam->streamWidth) / double(cam->calibWidth);
            //     cam->calibMat.at<double>(0,2)*=  double(cam->streamWidth) / double(cam->calibWidth);
            //     cam->calibMat.at<double>(1,1)*=  double(cam->streamWidth) / double(cam->calibWidth);
            //     cam->calibMat.at<double>(1,2)*=  double(cam->streamWidth) / double(cam->calibWidth);

            //     cv::initUndistortRectifyMap(cam->calibMat, cam->distCoeff, cv::Mat(), cam->calibMat, cv::Size(cam->streamWidth, cam->streamHeight), CV_32F, map1, map2);

            //     checkCuda( cudaMalloc(&d_input, distort.cols*distort.rows*distort.channels()*sizeof(uint8_t)) );
            //     checkCuda( cudaMalloc(&d_output, distort.cols*distort.rows*distort.channels()*sizeof(uint8_t)) );
            //     checkCuda( cudaMalloc(&d_map1, map1.cols*map1.rows*map1.channels()*sizeof(float)) );
            //     checkCuda( cudaMalloc(&d_map2, map2.cols*map2.rows*map2.channels()*sizeof(float)) );
            //     frame = distort.clone();

            //     checkCuda( cudaMemcpy(d_map1, (float*)map1.data,  map1.cols*map1.rows*map1.channels()*sizeof(float), cudaMemcpyHostToDevice));
            //     checkCuda( cudaMemcpy(d_map2, (float*)map2.data,  map2.cols*map2.rows*map2.channels()*sizeof(float), cudaMemcpyHostToDevice));
                
            //     first_iteration = false;
            // }
            // checkCuda( cudaMemcpy(d_input, (uint8_t*)distort.data,  distort.cols*distort.rows*distort.channels()*sizeof(uint8_t), cudaMemcpyHostToDevice));
            // remap(d_input, cam->streamWidth, cam->streamHeight, 3, d_map1, d_map2, d_output, cam->streamWidth , cam->streamHeight, 3);
            // checkCuda( cudaMemcpy((uint8_t*)frame.data , d_output, distort.cols*distort.rows*distort.channels()*sizeof(uint8_t), cudaMemcpyDeviceToHost));
            // // cv::remap(distort, frame, map1, map2, cv::INTER_LINEAR);
            // prof.tock("Undistort");

            //inference
            prof.tick("Inference");
            dnn_input.clear();
            dnn_input.push_back(frame.clone());  
            cam->detNN->update(dnn_input);
            detected= cam->detNN->detected;
            prof.tock("Inference");

            //feed the tracker
            // prof.tick("Tracker feeding");
            // cur_frame.clear();
            // for(auto d:detected){
            //     if(checkClass(d.cl, cam->dataset)){
            //         convertCameraPixelsToMapMeters((d.x + d.w / 2)*scale_x, (d.y + d.h)*scale_y, d.cl, *cam, north, east);
            //         tracking::obj_m obj;
            //         obj.frame   = 0;
            //         obj.cl      = d.cl;
            //         obj.x       = north;
            //         obj.y       = east;
            //         cur_frame.push_back(obj);
            //     }
            // }
            // t.track(cur_frame,tr_verbose);
            // prof.tock("Tracker feeding");

            //feed the viewer
            prof.tick("Viewer feeding");
            if(show && cam->show)
                viewer->setFrameData(frame, detected, void_lines, cam->id);
            prof.tock("Viewer feeding");

            // prof.tick("Prepare message");
            // //send the data if the message is not empty
            // prepareMessage(t, message, cam->geoConv, cam->id, cam->dataset);
            // if (!message.objects.empty()){
            //     communicator.send_message(&message, cam->portCommunicator);
            //     // std::cout<<"message sent!"<<std::endl;
            // }
            // prof.tock("Prepare message");   
        }
        prof.tock("Total time");   
        if (verbose) 
            prof.printStats();
    }

    // checkCuda( cudaFree(d_input));
    // checkCuda( cudaFree(d_output));
    // checkCuda( cudaFree(d_map1));
    // checkCuda( cudaFree(d_map2));
    return (void *)0;
}