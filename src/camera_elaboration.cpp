#include "camera_elaboration.h"






void *elaborateSingleCamera(void *ptr)
{
    edge::camera* cam = (edge::camera*) ptr;
    std::cout<<"Starting camera: "<< cam->id << std::endl;

    pthread_t video_cap;
    video_cap_data data;
    data.input = (char*)cam->input.c_str();

    if(cam->show)
        viewer->setClassesNames(cam->detNN->classesNames);
    
    if (pthread_create(&video_cap, NULL, readVideoCapture, (void *)&data)){
        fprintf(stderr, "Error creating thread\n");
        return (void *)1;
    }

    
    

    cv::Mat dnn_input;
    cv::Mat frame;
    std::vector<tk::dnn::box> detected;
    while(gRun){
        if(data.frame.data) {
            data.mtxF.lock();
            frame = data.frame;
            data.mtxF.unlock();
            dnn_input = frame.clone();    
            //inference
            cam->detNN->update(dnn_input);
            detected= cam->detNN->detected;
            if(cam->show)
                viewer->setFrameAndDetection(frame, detected);
        }
    }
    return (void *)0;
}