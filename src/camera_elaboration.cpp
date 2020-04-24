#include "camera_elaboration.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

std::ostream& operator<<(std::ostream& os, const edge::gps_obj& o){
    os<<std::setprecision(20)<<"LAT:\t"<<o.lat<<"\tLON:\t"<<o.lon<<"\tClass:\t"<<o.cl<<std::endl;
    return os;
}

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

edge::gps_obj convertCameraPixelToGPS(const int x, const int y, const int cl, const cv::Mat& prj_mat)
{
    edge::gps_obj o;
    o.cl = cl;

    //transform camera pixel into georeferenced map pixel
    std::vector<cv::Point2f> x_y, ll;
    x_y.push_back(cv::Point2f(x, y));
    cv::perspectiveTransform(x_y, ll, prj_mat);

    //tranform to map pixel into GPS
    pixel2GPS(ll[0].x, ll[0].y, o.lat, o.lon);

    return o;
}

void *elaborateSingleCamera(void *ptr)
{
    edge::camera* cam = (edge::camera*) ptr;
    std::cout<<"Starting camera: "<< cam->id << std::endl;

    pthread_t video_cap;
    video_cap_data data;
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
    int     age_threshold   = 0;
    bool    tr_verbose      = false;
    Tracking t(n_states, dt, initial_age, age_threshold);
    

    cv::Mat dnn_input;
    cv::Mat frame, distort;
    cv::Mat map1, map2;

    std::vector<cv::Point2f> map_pixels;
    std::vector<cv::Point2f> camera_pixels;

    std::vector<tk::dnn::box>   detected;
    std::vector<edge::gps_obj>        gps_objects;
    std::vector<Data>           cur_frame;
    std::vector<tracker_line>  lines;

    double north, east, up;
    double latitude, longitude, altitude;
    int map_pix_x, map_pix_y; 
    
    bool verbose = false;
    bool first_iteration = true;

    while(gRun){
        if(data.frame.data) {

            //copy the frame that the last frame read by the video capture thread
            data.mtxF.lock();
            distort = data.frame.clone();
            data.mtxF.unlock();
            
            // undistort 
            if (first_iteration){
                cv::initUndistortRectifyMap(cam->calibMat, cam->distCoeff, cv::Mat(), cam->calibMat, distort.size(), CV_16SC2, map1, map2);
                first_iteration = false;
            }
            frame = distort.clone();
            cv::remap(distort, frame, map1, map2, cv::INTER_CUBIC);

            //inference
            dnn_input = frame.clone();  
            cam->detNN->update(dnn_input);
            detected= cam->detNN->detected;
            

            gps_objects.clear();
            for(auto d:detected)
                if(d.cl < 6)
                    gps_objects.push_back(convertCameraPixelToGPS(d.x + d.w / 2, d.y + d.h, d.cl, cam->prjMat));
            
            if(verbose)
                for(auto o:gps_objects)
                    std::cout<<o;

            //feed the tracker
            cur_frame.clear();
            cur_frame.resize(gps_objects.size());
            for(int i=0; i<gps_objects.size(); ++i){
                //conversion from GPS to meters (needed for the tracker)
                geoConv.geodetic2Enu(gps_objects[i].lat, gps_objects[i].lon, 0, &north, &east, &up);
                cur_frame[i].frame_     = 0;
                cur_frame[i].class_     = gps_objects[i].cl;
                cur_frame[i].x_         = north;
                cur_frame[i].y_         = east;

                if(verbose)
                    std::cout<<"Frame: \t"<<cur_frame[i].frame_<<"\tCl: \t"<<cur_frame[i].class_<<"\tx: \t"<<cur_frame[i].x_<<"\ty: \t"<<cur_frame[i].y_<<std::endl;
            }
            t.Track(cur_frame,tr_verbose);

            //visualize the trackers
            lines.clear();
            for(auto tr: t.trackers_){
                if(tr.pred_list_.size()){
                    tracker_line line;

                    map_pixels.clear();
                    camera_pixels.clear();
                    for(int i=0; i < tr.pred_list_.size(); ++i){
                        //convert from meters to GPS
                        geoConv.enu2Geodetic(tr.pred_list_[i].x_, tr.pred_list_[i].y_, 0, &latitude, &longitude, &altitude);
                        //convert from GPS to map pixels
                        GPS2pixel(latitude, longitude, map_pix_x, map_pix_y);
                        map_pixels.push_back(cv::Point2f(map_pix_x, map_pix_y));

                        //add RoadUser to the message
                        if(i == tr.pred_list_.size() - 1 )
                            message.objects.push_back(getRoadUser(latitude, longitude, tr.pred_list_[i].vel_, tr.pred_list_[i].yaw_, tr.class_));
                    }

                    if(cam->show){
                        //transform map pixels to camers pixels
                        cv::perspectiveTransform(map_pixels, camera_pixels, cam->invPrjMat);
                        
                        //convert into viewer coordinates
                        for(auto cp: camera_pixels)
                        {
                            if(verbose)
                                std::cout<<"x:\t"<<cp.x<<"\t y:\t"<<cp.y<<std::endl;
                            line.points.push_back(viewer->convertPosition(cp.x, cp.y, -0.004, cam->id));
                        }
                        line.color = tk::gui::Color_t {tr.r_, tr.g_, tr.b_, 255};
                        lines.push_back(line);
                    }
                }
            }

            //feed the viewer
            if(cam->show)
                viewer->setFrameData(frame, detected, lines, cam->id);


            //prepare the message
            message.t_stamp_ms = getTimeMs();
            message.num_objects = message.objects.size();

            //send the data if there are any
            if (!message.objects.empty()){
                communicator.send_message(&message, cam->portCommunicator);
                std::cout<<"message sent!"<<std::endl;
            }
            message.objects.clear();

        }
    }
    return (void *)0;
}