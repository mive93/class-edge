#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <cstring>

#include "tkDNN/DetectionNN.h"
#include "EdgeViewer.h"

#define MAX_CAMERAS 10

namespace edge {
struct camera
{
    double adfGeoTransform[6];
    std::string input;
    std::string pmatrixPath;
    std::string maskfilePath;
    std::string cameraCalibPath;
    std::string maskFileOrientPath;
    tk::dnn::DetectionNN *detNN;  
    int id = 0;
    bool show = false;
};
}

std::ostream& operator<<(std::ostream& os, const edge::camera& c);

extern EdgeViewer *viewer;
extern bool gRun;


#endif /*DATA_H*/