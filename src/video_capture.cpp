#include "video_capture.h"

void *readVideoCapture( void *ptr )
{
    edge::video_cap_data* data = (edge::video_cap_data*) ptr;
    
    std::cout<<"Thread: "<<data->input<< " started" <<std::endl;
    cv::VideoCapture cap(data->input, cv::CAP_FFMPEG);
    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"Camera correctly started.\n";

    const int new_width     = data->width;
    const int new_height    = data->height;
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
        data->frame     = resized_frame.clone();
        data->mtxF.unlock();
        prof.tock("Frame copy");

        // prof.printStats();
    }
    return (void *)0;
}
