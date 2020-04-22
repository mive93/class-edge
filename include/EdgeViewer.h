#ifndef EDGEVIEWER_H
#define EDGEVIEWER_H

#include "tkCommon/gui/Viewer.h"

struct tracker_line
{
    std::vector<tk::common::Vector3<float>> points;
    tk::gui::Color_t                        color;
};

class EdgeViewer : public tk::gui::Viewer {
    private:
        
        GLuint frameTexture;
        cv::Mat frame;
        bool newFrame = false;
        std::mutex mtxNewFrame;
        
        int frame_width, frame_height;
        float xScale, yScale;
        float aspectRatio;
        
        std::vector<tk::dnn::box> detected;        
        std::vector<tracker_line> lines;
        
        std::vector<std::string> classesNames;
        std::vector<tk::gui::Color_t> colors;

        tk::common::Vector3<float> pose;
        tk::common::Vector3<float> size;

        
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
                frame_width = frame.cols;
                frame_height = frame.rows;
                aspectRatio = (float)frame_width/(float)frame_height; 
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
                xScale = xLim;
                yScale = xScale*aspectRatio;
                tkDrawTexture(frameTexture, xScale, yScale);
            } glPopMatrix();

            //draw detections
            for(auto d: detected){
                tk::gui::Color_t col = tk::gui::color::RED;
                if(colors.size() > 0) col = colors[d.cl];                    
                tkSetColor(col);

                pose = convertPosition((d.x+d.w/2), (d.y+d.h/2), -0.002);
                size = convertSize(d.w, d.h);
                tkDrawRectangle(pose, size, false);

                tkDrawText(classesNames[d.cl],tk::common::Vector3<float>{pose.x - d.w/2.0/frame_width * yScale, pose.y+ d.h/2.0/frame_height*xScale, pose.z},
                                tk::common::Vector3<float>{0, 0, 0},
                                tk::common::Vector3<float>{0.03*xScale, 0.03*yScale, 0});
            }

            for(auto l:lines)
            {
                tkSetColor(l.color);
                tkDrawLine(l.points);
            }
            
        }
        void setFrameData(const cv::Mat &new_frame, const std::vector<tk::dnn::box> &new_detected, const std::vector<tracker_line>& new_lines){ 
            frame = new_frame;
            detected = new_detected;
            lines = new_lines;
            
            mtxNewFrame.lock();
            newFrame = true;
            mtxNewFrame.unlock();
        }

        void setClassesNames(const std::vector<std::string>& classes_names){
            classesNames = classes_names;
        }

        tk::common::Vector3<float> convertPosition(int x, int y, float z){
            float new_x = ((float)x/(float)frame_width - 0.5)*yScale;
            float new_y = -((float)y/(float)frame_height -0.5)*xScale;
            return tk::common::Vector3<float>{new_x, new_y, z};
        }

        tk::common::Vector3<float> convertSize(int w, int h){
            float new_w = (float)w/(float)frame_width * yScale;
            float new_h = (float)h/(float)frame_height * xScale;
            return tk::common::Vector3<float>{new_w, new_h, 0};
        }

        void setColors(const int n_classes){
            for(int c=0; c<n_classes; c++) {
                int offset = c*123457 % n_classes;
                float r = getColor(2, offset, n_classes);
                float g = getColor(1, offset, n_classes);
                float b = getColor(0, offset, n_classes);
                colors.push_back(tk::gui::Color_t {int(255.0*b), int(255.0*g), int(255.0*r), 255});
            }

        }
};

#endif /*EDGEVIEWER_H*/