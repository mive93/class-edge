#include <iostream>
#include "configuration.h"
#include <yaml-cpp/yaml.h>
#include "tkCommon/exceptions.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include "camera_elaboration.h"
#include "data.h"

void sig_handler(int signo) {
    std::cout<<"request stop\n";
    gRun = false;
}

int main(int argc, char **argv)
{
    tk::exceptions::handleSegfault();
    std::vector<edge::camera> cameras;
    std::string net, tif_map_path;
    char type;
    int n_classes;
    readParameters(argc, argv, cameras, net, type, n_classes, tif_map_path);
    initializeCamerasNetworks(cameras, net, type, n_classes);

    pthread_t threads[MAX_CAMERAS];
    int iret[MAX_CAMERAS];
    for(size_t i=0; i<cameras.size(); ++i)
        iret[i] = pthread_create( &threads[i], NULL, elaborateSingleCamera, (void*) &cameras[i]);

    viewer = new EdgeViewer();
    viewer->setWindowName("Cameras");
    viewer->setBackground(tk::gui::color::DARK_GRAY);
    viewer->initOnThread();

    viewer->joinThread();

    for(size_t i=0; i<cameras.size(); ++i)
        pthread_join( threads[i], NULL);

    for(size_t i=0; i<cameras.size(); ++i)
        printf("Thread %d returns: %d\n", i,iret[i]); 
    
    return EXIT_SUCCESS;
}
