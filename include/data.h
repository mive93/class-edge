#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <cstring>
#include <iomanip>
#include <fstream>

#include "tkCommon/geodetic_conv.h"
#include "tkDNN/DetectionNN.h"
#include "EdgeViewer.h"

#define MAX_CAMERAS 10

namespace edge {
struct camera_params
{
    std::string             input = "";
    std::string             pmatrixPath = "";
    std::string             maskfilePath = "";
    std::string             cameraCalibPath = "";
    std::string             maskFileOrientPath = "";
    int                     id = 0;
    int                     streamWidth = 0;
    int                     streamHeight = 0;
    bool                    show = false;    
};

enum Dataset_t { BDD, COCO, VOC};

struct camera{
    cv::Mat                         prjMat;
    cv::Mat                         invPrjMat;
    cv::Mat                         calibMat;
    cv::Mat                         distCoeff;
    std::string                     input = "";
    std::string                     ipCommunicator = "127.0.0.1";
    tk::dnn::DetectionNN*           detNN = nullptr;  
    tk::common::GeodeticConverter   geoConv; 
    double*                         adfGeoTransform = nullptr;
    int                             id = 0;
    int                             portCommunicator = 8888;
    int                             calibWidth;
    int                             calibHeight;
    int                             streamWidth;
    int                             streamHeight;
    Dataset_t                       dataset;
    bool                            show = false;
    
};
}

std::ostream& operator<<(std::ostream& os, const edge::camera_params& c);
std::ostream& operator<<(std::ostream& os, const edge::camera& c);

extern edge::EdgeViewer *viewer;
extern bool gRun;
extern bool show;

// #define COCO
#define BERKELEY

//COCO
#ifdef COCO
#define CAR_ID 2
#define TRUCK_ID 7
#endif

//Berkeley
#ifdef BERKELEY
#define CAR_ID 1
#define TRUCK_ID 2
#endif

//AICityChallenge
#define CAR_ID_AICC 1
#define TRUCK_ID_AICC 2

#endif /*DATA_H*/