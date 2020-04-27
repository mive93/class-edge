#ifndef EDGEVIEWER_H
#define EDGEVIEWER_H

#include "tkCommon/gui/Viewer.h"
#include <map>

namespace edge{

struct tracker_line{
    std::vector<tk::common::Vector3<float>> points;
    tk::gui::Color_t                        color;
};

struct camera_data{
    std::vector<tk::dnn::box> detected;        
    std::vector<tracker_line> lines;
    cv::Mat frame;
    int id = 0;
    bool newFrame = false;
    std::mutex *mtxNewFrame;

    GLuint frameTexture;
        
    int frame_width, frame_height;
    float xScale, yScale;
    float aspectRatio;

    tk::common::Vector3<float> pose;
    tk::common::Vector3<float> size;
};

#define MAX_CAMERA_VIZ 6

class EdgeViewer : public tk::gui::Viewer {
    private:
        
        std::vector<std::string> classesNames;
        std::vector<tk::gui::Color_t> colors;

        int nCameras = 1;
        int cameraIndex = 0;
        std::mutex mtxIndex;

        std::map<int,int> idIndexBind;

        std::vector<camera_data> cameraData;

        void settkViewport2D(const int index){
            int viewport_h = height;
            int viewport_w = width;
            int viewport_x = 0;
            int viewport_y = 0;
            if(nCameras == 2)
            {
                viewport_h /= 2;
                viewport_y = viewport_h*index;
            }
            else if(nCameras == 3 || nCameras == 4){
                viewport_w /= 2;
                viewport_h /= 2;
                viewport_x = viewport_w*(index/2);
                viewport_y = viewport_h*(index%2);
            }
            else if(nCameras == 5 || nCameras == 6){
                viewport_w /= 2;
                viewport_h /= 3;
                viewport_x = viewport_w*(index/2);
                viewport_y = viewport_h*(index%3);
            }

            tkViewport2D(viewport_w, viewport_h, viewport_x , viewport_y);
        }

        
    public:
        EdgeViewer(int n_cameras) {
            if(n_cameras < MAX_CAMERA_VIZ)
                nCameras = n_cameras;
            else
                std::cout<<"Maximum "<<MAX_CAMERA_VIZ<<" cameras can be shown"<<std::endl;
            cameraData.resize(nCameras);
            for(auto& cd: cameraData)
                cd.mtxNewFrame = new std::mutex;
        }
        ~EdgeViewer() {
            // for(auto& cd: cameraData)
            //     delete cd.mtxNewFrame;
        }

        void init() {
            tk::gui::Viewer::init();
            for(auto& cd: cameraData)
                glGenTextures(1, &cd.frameTexture);
        }

        void draw() {
            tk::gui::Viewer::draw();

            for(auto& cd:cameraData){
            
                if(cd.newFrame){
                    //when receiving a new frame, convert cv::Mat into GLuint texture
                    glBindTexture(GL_TEXTURE_2D, cd.frameTexture);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,cd.frame.cols, cd.frame.rows, 0, GL_BGR,GL_UNSIGNED_BYTE, cd.frame.data);
                    cd.frame_width = cd.frame.cols;
                    cd.frame_height = cd.frame.rows;
                    cd.aspectRatio = float(cd.frame_width)/float(cd.frame_height); 
                    cd.mtxNewFrame->lock();
                    cd.newFrame = false;
                    cd.mtxNewFrame->unlock();
                }
                //set 2D view 
                // tkViewport2D(width, height/2., 0 , height/2.);
                settkViewport2D(idIndexBind[cd.id]);

                //draw frame
                glPushMatrix(); {
                    glTranslatef(0, 0, 0.001);
                    glColor4f(1,1,1,1);
                    cd.xScale = xLim;
                    cd.yScale = cd.xScale*cd.aspectRatio;
                    tkDrawTexture(cd.frameTexture, cd.xScale, cd.yScale);
                } glPopMatrix();

                //draw detections
                for(const auto& d: cd.detected){
                    tk::gui::Color_t col = tk::gui::color::RED;
                    if(colors.size() > 0) col = colors[d.cl];                    
                    tkSetColor(col);

                    cd.pose = convertPosition((d.x+d.w/2), (d.y+d.h/2), -0.002, cd.id);
                    cd.size = convertSize(d.w, d.h, cd.id);
                    tkDrawRectangle(cd.pose, cd.size, false);

                    tkDrawText(classesNames[d.cl],tk::common::Vector3<float>{cd.pose.x - d.w/2.0/cd.frame_width * cd.yScale, cd.pose.y+ d.h/2.0/cd.frame_height*cd.xScale, cd.pose.z},
                                    tk::common::Vector3<float>{0, 0, 0},
                                    tk::common::Vector3<float>{0.03*cd.xScale, 0.03*cd.yScale, 0});
                }

                for(const auto& l:cd.lines){
                    tkSetColor(l.color);
                    tkDrawLine(l.points);
                }
            
            }
            
        }
        void setFrameData(const cv::Mat &new_frame, const std::vector<tk::dnn::box> &new_detected, const std::vector<tracker_line>& new_lines, int id){ 
            int cur_index = idIndexBind[id];
            cameraData[cur_index].frame = new_frame.clone();
            cameraData[cur_index].detected = new_detected;
            cameraData[cur_index].lines = new_lines;
            
            cameraData[cur_index].mtxNewFrame->lock();
            cameraData[cur_index].newFrame = true;
            cameraData[cur_index].mtxNewFrame->unlock();
        }

        void setClassesNames(const std::vector<std::string>& classes_names){
            classesNames = classes_names;
        }

        tk::common::Vector3<float> convertPosition(int x, int y, float z, int id){
            float new_x = ((float)x/(float)cameraData[idIndexBind[id]].frame_width - 0.5)*cameraData[idIndexBind[id]].yScale;
            float new_y = -((float)y/(float)cameraData[idIndexBind[id]].frame_height -0.5)*cameraData[idIndexBind[id]].xScale;
            return tk::common::Vector3<float>{new_x, new_y, z};
        }

        tk::common::Vector3<float> convertSize(int w, int h, int id){
            float new_w = (float)w/(float)cameraData[idIndexBind[id]].frame_width * cameraData[idIndexBind[id]].yScale;
            float new_h = (float)h/(float)cameraData[idIndexBind[id]].frame_height * cameraData[idIndexBind[id]].xScale;
            return tk::common::Vector3<float>{new_w, new_h, 0};
        }

        void setColors(const int n_classes){
            for(int c=0; c<n_classes; ++c) {
                int offset = c*123457 % n_classes;
                float r = getColor(2, offset, n_classes);
                float g = getColor(1, offset, n_classes);
                float b = getColor(0, offset, n_classes);
                colors.push_back(tk::gui::Color_t {int(255.0*b), int(255.0*g), int(255.0*r), 255});
            }
        }

        void setNCameras(const int n_cameras){
            nCameras = n_cameras;
        }

        void bindCamera(const int id)
        {         
            mtxIndex.lock();
            idIndexBind[id] = cameraIndex++;
            mtxIndex.unlock();
            cameraData[idIndexBind[id]].id = id;
        }
};

}

#endif /*EDGEVIEWER_H*/