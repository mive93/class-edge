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

void convertCameraPixelsToMapMeters(const int x, const int y, const int cl, edge::camera& cam, double& north, double& east);
std::vector<edge::tracker_line> getTrackingLines(const tracking::Tracking& t, edge::camera& cam,const float scale_x=1, const float scale_y=1, bool verbose=false);

void prepareMessage(const tracking::Tracking& t, MasaMessage& message,tk::common::GeodeticConverter& geoConv, 
                    const int cam_id, edge::Dataset_t dataset, uint64_t t_stamp_acquisition_ms);

void *elaborateSingleCamera(void *ptr);

#endif /*CAMERAELABORATION_H*/