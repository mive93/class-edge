#ifndef EDGEVIEWER_H
#define EDGEVIEWER_H


#include "tkCommon/gui/Viewer.h"
class EdgeViewer : public tk::gui::Viewer {
    private:
        
        GLuint frameTexture;
        cv::Mat frame;
        bool newFrame = false;
        std::mutex mtxNewFrame;
        
        int width, height;
        float xScale, yScale;
        float aspectRatio;
        
        std::vector<tk::dnn::box> detected;        
        
        std::vector<std::string> classesNames;

        tk::common::Vector3<float> pose;
        tk::common::Vector3<float> size;

        tk::common::Vector3<float> convertPosition(tk::dnn::box b, float z){
            float x = ((b.x+b.w/2.0)/width - 0.5)*yScale;
            float y = -((b.y+b.h/2)/height -0.5)*xScale;
            return tk::common::Vector3<float>{x, y, z};
        }

        tk::common::Vector3<float> convertSize(tk::dnn::box b){
            float w = b.w/width * yScale;
            float h = b.h/height * xScale;
            return tk::common::Vector3<float>{w, h, 0};
        }
    public:
        EdgeViewer() {}
        ~EdgeViewer() {}

        void init() {
            tk::gui::Viewer::init();
            glGenTextures(1, &frameTexture);
        }

        void draw() {
            tk::gui::Viewer::draw();
            
            if(newFrame){
                //when receiving a new frame, convert cv::Mat into GLuint texture
                glBindTexture(GL_TEXTURE_2D, frameTexture);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,frame.cols, frame.rows, 0, GL_BGR,GL_UNSIGNED_BYTE, frame.data);
                width = frame.cols;
                height = frame.rows;
                aspectRatio = (float)width/(float)height; 
                mtxNewFrame.lock();
                newFrame = false;
                mtxNewFrame.unlock();
            }
            //set 2D view
            tkViewport2D(width, height);

            //draw frame
            glPushMatrix(); {
                glTranslatef(0, 0, 0.001);
                glColor4f(1,1,1,1);
                xScale = 1;
                yScale = xScale*aspectRatio;
                tkDrawTexture(frameTexture, xScale, yScale);
            } glPopMatrix();

            //draw detections
            for(auto d: detected){   
                tk::gui::Color_t col = tk::gui::color::RED;
                tkSetColor(col);
                pose = convertPosition(d, -0.002);
                size = convertSize(d);

                tkDrawRectangle(pose, size, false);
                tkDrawText(classesNames[d.cl],tk::common::Vector3<float>{pose.x - d.w/2.0/width * yScale, pose.y+ d.h/2.0/height*xScale, pose.z},
                                tk::common::Vector3<float>{0, 0, 0},
                                tk::common::Vector3<float>{0.03*xScale, 0.03*yScale, 0});
            }
            
        }
        void setFrameAndDetection(const cv::Mat &new_frame, const std::vector<tk::dnn::box> &new_detected){ 
            frame = new_frame;
            detected = new_detected;
            mtxNewFrame.lock();
            newFrame = true;
            mtxNewFrame.unlock();
        }

        void setClassesNames(const std::vector<std::string>& classes_names){
            classesNames = classes_names;
        }
};

#endif /*EDGEVIEWER_H*/