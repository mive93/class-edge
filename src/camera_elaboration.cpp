#include "camera_elaboration.h"



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

    if(cam->show){
        viewer->setClassesNames(cam->detNN->classesNames);
        viewer->setColors(cam->detNN->classes);
    }
    
    if (pthread_create(&video_cap, NULL, readVideoCapture, (void *)&data)){
        fprintf(stderr, "Error creating thread\n");
        return (void *)1;
    }
    
    //TODO instantiate the socket

    //initiate the tracker
    float   dt              = 0.03;
    int     n_states        = 5;
    int     initial_age     = 15;
    int     age_threshold   = 0;
    bool    tr_verbose      = false;
    Tracking t(n_states, dt, initial_age, age_threshold);
    

    cv::Mat dnn_input;
    cv::Mat frame;

    std::vector<cv::Point2f> map_pixels;
    std::vector<cv::Point2f> camera_pixels;

    std::vector<tk::dnn::box>   detected;
    std::vector<edge::gps_obj>        gps_objects;
    std::vector<Data>           cur_frame;
    std::vector<tracker_line>  lines;

    
    
    double north, east, up;
    double latitude, longitude, altitude;
    int map_pix_x, map_pix_y; 
    int frame_nbr = 0; //FIXME
    bool verbose = false;


    while(gRun){
        if(data.frame.data) {
            data.mtxF.lock();
            frame = data.frame;
            data.mtxF.unlock();
            
            frame_nbr++;
            dnn_input = frame.clone();  

            //TODO undistort

            //inference
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
                cur_frame[i].frame_     = frame_nbr;
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
                    line.color = tk::gui::Color_t {tr.r_, tr.g_, tr.b_, 255};

                    map_pixels.clear();
                    camera_pixels.clear();
                    for(auto traj: tr.pred_list_){
                        //convert from meters to GPS
                        geoConv.enu2Geodetic(traj.x_, traj.y_, 0, &latitude, &longitude, &altitude);
                        //convert from GPS to map pixels
                        GPS2pixel(latitude, longitude, map_pix_x, map_pix_y);
                        map_pixels.push_back(cv::Point2f(map_pix_x, map_pix_y));
                    }

                    //transform map pixels to camers pixels
                    cv::perspectiveTransform(map_pixels, camera_pixels, cam->invPrjMat);
                    
                    //convert into viewer coordinates
                    for(auto cp: camera_pixels)
                    {
                        if(verbose)
                            std::cout<<"x:\t"<<cp.x<<"\t y:\t"<<cp.y<<std::endl;
                        line.points.push_back(viewer->convertPosition(cp.x, cp.y, -0.004));
                    }

                    lines.push_back(line);
                }
            }

            if(cam->show)
                viewer->setFrameData(frame, detected, lines);

            //TODO send the messages

        }
    }
    return (void *)0;
}