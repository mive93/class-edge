#include "video_capture.h"

void *readVideoCapture( void *ptr )
{
    edge::video_cap_data* data = (edge::video_cap_data*) ptr;
    
    std::cout<<"thread: "<<data->input<<std::endl;
    cv::VideoCapture cap(data->input);
    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";

    const int new_width     = 960;
    const int new_height    = 540;
    cv::Mat frame, resized_frame;

    edge::Profiler prof("Video capture" + std::string(data->input));
    while(gRun) {
        prof.tick("Frame acquisition");
        cap >> frame; 
        prof.tock("Frame acquisition");
        if(!frame.data) {
            usleep(1000000);
            cap.open(data->input);
            printf("cap reinitialize\n");
            continue;
        } 
        
        //resizing the image to 960x540 (the DNN takes in input 544x320)
        prof.tick("Frame resize");
        cv::resize (frame, resized_frame, cv::Size(new_width, new_height)); 
        prof.tock("Frame resize");

        prof.tick("Frame copy");
        data->mtxF.lock();
        data->oWidth    = frame.cols;
        data->oHeight   = frame.rows;
        data->width     = new_width;
        data->height    = new_height;
        data->frame     = resized_frame.clone();
        data->mtxF.unlock();
        prof.tock("Frame copy");

        prof.printStats();
    }
    return (void *)0;
}
