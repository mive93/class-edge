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

void pixel2coord(const int x, const int y, double &lat, double &lon)
{
    //conversion from pixels to GPS, via georeferenced map parameters
    double xoff, a, b, yoff, d, e;
    xoff    = adfGeoTransform[0];
    a       = adfGeoTransform[1];
    b       = adfGeoTransform[2];
    yoff    = adfGeoTransform[3];
    d       = adfGeoTransform[4];
    e       = adfGeoTransform[5];

    lon     = a * x + b * y + xoff;
    lat     = d * x + e * y + yoff;
}
void coord2pixel(double lat, double lon, int &x, int &y)
{
    //conversion from GPS to pixels, via georeferenced map parameters
    x = int(round( (lon - adfGeoTransform[0]) / adfGeoTransform[1]) );
    y = int(round( (lat - adfGeoTransform[3]) / adfGeoTransform[5]) );
}

int main(int argc, char **argv)
{
    tk::exceptions::handleSegfault();
    signal(SIGINT, sig_handler);    

    std::vector<edge::camera> cameras = configure(argc, argv);
    
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

    free(adfGeoTransform);
    
    return EXIT_SUCCESS;
}
