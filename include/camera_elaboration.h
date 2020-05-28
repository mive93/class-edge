#ifndef CAMERAELABORATION_H
#define CAMERAELABORATION_H

#include <pthread.h>
#include <vector> 
#include <fstream>
#include <iostream>

#include "Tracking.h"

#include"video_capture.h"
#include"message.h"


void pixel2GPS(const int x, const int y, double &lat, double &lon, double* adfGeoTransform);
void GPS2pixel(double lat, double lon, int &x, int &y, double* adfGeoTransform);

void convertCameraPixelsToMapMeters(const int x, const int y, const int cl, const cv::Mat& prj_mat, double& north, double& east, tk::common::GeodeticConverter geo_conv, double* adf_geo_transform=nullptr);
std::vector<edge::tracker_line> getTrackingLines(const tracking::Tracking& t, const cv::Mat& inv_prj_mat, tk::common::GeodeticConverter geo_conv, std::vector<tk::dnn::box>&  detected, double*  adf_geo_transform=nullptr, const int cam_id=0,const float scale_x=1, const float scale_y=1, bool verbose=false, std::ofstream *det_out=nullptr, const int frame_id=0);

void prepareMessage(const tracking::Tracking& t, MasaMessage& message,tk::common::GeodeticConverter& geoConv, const int cam_id, edge::Dataset_t dataset);

void *elaborateSingleCamera(void *ptr);

#endif /*CAMERAELABORATION_H*/