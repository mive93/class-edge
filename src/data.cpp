#include "data.h"


std::ostream& operator<<(std::ostream& os, const edge::camera& c){
    os << c.id << '\t' << c.input << '\t' << c.pmatrixPath<< '\t' << c.cameraCalibPath;
    return os;
}

EdgeViewer *viewer = nullptr;
bool gRun = true;