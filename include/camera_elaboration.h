#ifndef CAMERAELABORATION_H
#define CAMERAELABORATION_H

#include <pthread.h>
#include <vector> 
#include <fstream>
#include <iostream>
#include"video_capture.h"

void *elaborateSingleCamera(void *ptr);

#endif /*CAMERAELABORATION_H*/