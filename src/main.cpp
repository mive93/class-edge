#include <iostream>
#include "configuration.h"

int main(int argc, char **argv)
{
    std::vector<edge::camera> cameras;
    std::string net = "yolo3_berkeley_fp32.rt";
    char type = 'y';
    int n_classes = 80;
    readParameters(argc, argv, cameras, net, type, n_classes);
    initializeCamerasNetworks(cameras, net, type, n_classes);

    return EXIT_SUCCESS;
}
