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
    
    cv::Mat frame;
    while(gRun) {
        cap >> frame; 
        if(!frame.data) {
            usleep(1000000);
            cap.open(data->input);
            printf("cap reinitialize\n");
            continue;
        } 
        data->mtxF.lock();
        data->frame = frame.clone();
        data->mtxF.unlock();
    }
    return (void *)0;
}
