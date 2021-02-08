#ifndef VIDEOCAPTURE_H
#define VIDEOCAPTURE_H

#include <pthread.h>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>


#include "tkDNN/utils.h"
#include "data.h"
#include "Profiler.h"
#include "message.h"

namespace edge{

struct video_cap_data
{
    char* input     = nullptr;
    cv::Mat frame;
    std::mutex mtxF;
    int width   = 0;
    int height  = 0;
    int camId   = 0;
    bool record = false;
};
}

void *readVideoCapture(void *ptr);

#endif /*VIDEOCAPTURE_H*/