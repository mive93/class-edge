#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <iostream>
#include "tkDNN/utils.h"
#include "tkDNN/CenternetDetection.h"
#include "tkDNN/MobilenetDetection.h"
#include "tkDNN/Yolo3Detection.h"
#include <yaml-cpp/yaml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"

std::string executeCommandAndGetOutput(const char * command);
std::string decryptString(std::string encrypted, const std::string& password);
std::string encryptString(std::string to_encrypt, const std::string& password);

bool readParameters(int argc, char **argv,std:: vector<edge::camera>& cameras,std::string& net, char& type, int& n_classes, std::string& tif_map_path);
bool readParameters(int argc, char **argv,std:: vector<edge::camera>& cameras,std::string& net, char& type, int& n_classes, std::string& tif_map_path);
void initializeCamerasNetworks(std:: vector<edge::camera>& cameras, const std::string& net, const char type, int& n_classes);

#endif /*CONFIGURATION_H*/