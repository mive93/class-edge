#include "data.h"


std::ostream& operator<<(std::ostream& os, const edge::camera_params& c){
    
    os<<"----------------------------------------------------\n";
    os<< "id \t\t\t" << c.id <<std::endl;
    os<< "input \t\t\t" << c.input<<std::endl;
    os<< "pmatrixPath \t\t" << c.pmatrixPath <<std::endl;
    os<< "maskfilePath \t\t" << c.maskfilePath <<std::endl;
    os<< "cameraCalibPath \t" << c.cameraCalibPath <<std::endl;
    os<< "maskFileOrientPath \t" << c.maskFileOrientPath <<std::endl;
    os<< "show \t\t\t" << (int)c.show <<std::endl;
    os<<"----------------------------------------------------\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const edge::camera& c){

    os<<"----------------------------------------------------\n";
    os<< "id \t\t\t" << c.id <<std::endl;
    os<< "input \t\t\t" << c.input<<std::endl;
    os<< "detNN \t\t\t" << c.detNN<<std::endl;
    os<< "show \t\t\t" << (int)c.show <<std::endl;
    os<< "distCoeff \t\t" << c.distCoeff <<std::endl;
    os<< "calibMat: \n" << c.calibMat <<std::endl;
    os<< "prjMat: \n" << c.prjMat <<std::endl;
    os<< "invPrjMat: \n" << c.invPrjMat <<std::endl;
    os<<"----------------------------------------------------\n";

}

EdgeViewer *viewer = nullptr;
bool gRun = true;
double *adfGeoTransform = nullptr;
tk::common::GeodeticConverter geoConv;