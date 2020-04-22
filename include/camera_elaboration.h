#ifndef CAMERAELABORATION_H
#define CAMERAELABORATION_H

#include <pthread.h>
#include <vector> 
#include <fstream>
#include <iostream>
#include"video_capture.h"
#include "Tracking.h"



namespace edge { 
    struct gps_obj
    {
        double lat  = 0;
        double lon  = 0;
        int    cl   = 0;
    };
}

std::ostream& operator<<(std::ostream& os, const edge::gps_obj& o);

void pixel2GPS(const int x, const int y, double &lat, double &lon);
void GPS2pixel(double lat, double lon, int &x, int &y);
edge::gps_obj convertCameraPixelToGPS(const int x, const int y, const int cl, const cv::Mat& prj_mat);

void *elaborateSingleCamera(void *ptr);

#endif /*CAMERAELABORATION_H*/