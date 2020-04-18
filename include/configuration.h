#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <iostream>
#include "utils.h"
#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"
#include <yaml-cpp/yaml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

std::string executeCommandAndGetOutput(const char * command);
std::string decryptString(std::string encrypted, const std::string& password);
std::string encryptString(std::string to_encrypt, const std::string& password);

bool readParameters(int argc, char **argv,std:: vector<edge::camera>& cameras,std::string& net, char& type, int& n_classes, std::string& tif_map_path);
void initializeCamerasNetworks(std:: vector<edge::camera>& cameras, const std::string& net, const char type, int& n_classes);

#endif /*CONFIGURATION_H*/