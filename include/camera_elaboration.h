#ifndef CAMERAELABORATION_H
#define CAMERAELABORATION_H

#include <pthread.h>
#include <vector> 
#include <fstream>
#include <iostream>

#include "Tracking.h"

#include"video_capture.h"
#include"message.h"


void pixel2GPS(const int x, const int y, double &lat, double &lon);
void GPS2pixel(double lat, double lon, int &x, int &y);

void convertCameraPixelsToMapMeters(const int x, const int y, const int cl, const cv::Mat& prj_mat, double& north, double& east);
std::vector<edge::tracker_line> getTrackingLines(const tracking::Tracking& t, const cv::Mat& inv_prj_mat, const int cam_id, bool verbose=false);

void prepareMessage(const tracking::Tracking& t, MasaMessage& message);

void *elaborateSingleCamera(void *ptr);

#endif /*CAMERAELABORATION_H*/