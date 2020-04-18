#include <iostream>
#include "configuration.h"
#include <yaml-cpp/yaml.h>

int main(int argc, char **argv)
{
    std::vector<edge::camera> cameras;
    std::string net = "yolo3_berkeley_fp32.rt";
    std::string tif_map_path = "../data/masa_map.tif";
    char type = 'y';
    int n_classes = 80;
    readParameters(argc, argv, cameras, net, type, n_classes, tif_map_path);
    initializeCamerasNetworks(cameras, net, type, n_classes);

    return EXIT_SUCCESS;
}
