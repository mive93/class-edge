#include <iostream>

#include "utils.h"
#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"
#include <yaml-cpp/yaml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace edge {

struct camera
{
    double adfGeoTransform[6];
    std::string input;
    std::string pmatrixPath;
    std::string maskfilePath;
    std::string cameraCalibPath;
    std::string maskFileOrientPath;
    tk::dnn::DetectionNN *detNN;  
    int id = 0;
    bool show = false;
};

std::ostream& operator<<(std::ostream& os, const camera& c)
{
    os << c.id << '\t' << c.input << '\t' << c.pmatrixPath<< '\t' << c.cameraCalibPath;
    return os;
}

}

void initializeCamerasNetworks(std:: vector<edge::camera>& cameras, const std::string& net, const char type, int& n_classes){
    for(auto &c: cameras){
        switch(type){
            case 'y':
                c.detNN = new tk::dnn::Yolo3Detection();
                break;
            case 'c':
                c.detNN = new tk::dnn::CenternetDetection();
                break;
            case 'm':
                c.detNN = new tk::dnn::MobilenetDetection();
                n_classes++;
                break;
            default:
            FatalError("Network type not allowed\n");
        }
        c.detNN->init(net, n_classes);
    }
}

std::string decryptString(std::string encrypted){
    FILE *fpipe;
    std::string command = "echo '"+ encrypted +"' | openssl enc -e -aes-256-cbc -a -d -salt -iter 100000";
    char c = 0;

    if (0 == (fpipe = (FILE*)popen(command.c_str(), "r")))
        FatalError("popen() failed.");

    std::string decrypted; 
    while (fread(&c, sizeof c, 1, fpipe))
        decrypted.push_back(c);
    std::cout<<decrypted<<std::endl;

    pclose(fpipe);
    return decrypted;
}

bool readParameters(int argc, char **argv,std:: vector<edge::camera>& cameras,std::string& net, char& type, int& n_classes){
    std::string help =  "class-edge demo\nCommand:\n"
                        "-i\tparameters file\n"
                        "-n\tnetwork rt path\n"
                        "-t\ttype of network (only y|c|m admissed)\n"
                        "-c\t number of classes for the network\n"
                        "\tlist of camera ids (n ids expected)\n\n";

    //default values
    std::string params_path = "";
    
    //read arguments
    for(int opt;(opt = getopt(argc, argv, ":i:h:n:c:t:")) != -1;)
    {
        switch(opt)
        {
            case 'h':
                std::cout<<help<<std::endl;
                return false;
            case 'i':
                params_path = optarg;
                std::cout<<"Input parameters file in use: "<<params_path<<std::endl;
                break;
            case 'c':
                n_classes = atoi(optarg);
                std::cout<<"Number of classes specified: "<<n_classes<<std::endl;
                break;
            case 'n':
                net = optarg;
                std::cout<<"Network rt to use: "<<net<<std::endl;
                break;
            case 't':
                type = optarg[0];
                std::cout<<"Type of network in use: "<<type<<std::endl;
                if(type != 'y' && type != 'm' && type != 'c')
                FatalError("Unknown type of network, only y|c|m admitted");
                break;
            case ':':
                FatalError("This option needs a value");
                break;
            case '?':
                printf("You have digited: -%c\t", optopt);
                FatalError("Unknown option");
                break;
        }
    }

    //look for extra arguments (the ids of the cameras)
    std::vector<int> cameras_ids;
    for(; optind < argc; optind++)
        cameras_ids.push_back(atoi(argv[optind]));

    std::cout<<cameras_ids.size()<<" camera ids given: ";
    for(size_t i; i < cameras_ids.size(); ++i)
        std::cout<<cameras_ids[i]<<" ";
    std::cout<<std::endl;

    //if no parameters file given, set all default values
    if(params_path == "") {
        net = "yolo3_berkeley_fp32.rt";
        cameras.resize(1);
        cameras[0].id = 20936;
        cameras[0].input = "../data/20936.mp4";
        cameras[0].pmatrixPath = "../data/pmat_new/pmat_07-03-20936_20p.txt";
        cameras[0].maskfilePath = "../data/masks/20936_mask.jpg";
        cameras[0].cameraCalibPath = "../data/calib_cameras/20936.params";
        cameras[0].maskFileOrientPath = "../data/masks_orient/1920-1080_mask_null.jpg";
        cameras[0].show = false;
    }
    else{
        YAML::Node config   = YAML::LoadFile(params_path);
        net     = config["net"].as<std::string>();
        type    = config["type"].as<char>();
        n_classes = config["classes"].as<int>();
        
        YAML::Node cameras_yaml = config["cameras"];
        bool use_info;
        int n_cameras = 0;
        for(int i=0; i<cameras_yaml.size(); i++)
        {
            int camera_id = cameras_yaml[i]["id"].as<int>();

            //save infos only of the cameras whose ids where passed as args
            use_info = false;
            for(auto id: cameras_ids)
                if(camera_id == id)
                    use_info = true;
            if(!use_info) continue;

            cameras.resize(++n_cameras);
            cameras[n_cameras-1].id                 = camera_id;
            if (cameras_yaml[i]["encrypted"].as<int>())
                cameras[n_cameras-1].input          = decryptString(cameras_yaml[i]["input"].as<std::string>());
            else            
                cameras[n_cameras-1].input          = cameras_yaml[i]["input"].as<std::string>();
            cameras[n_cameras-1].pmatrixPath        = cameras_yaml[i]["pmatrix"].as<std::string>();
            cameras[n_cameras-1].maskfilePath       = cameras_yaml[i]["maskfile"].as<std::string>();
            cameras[n_cameras-1].cameraCalibPath    = cameras_yaml[i]["cameraCalib"].as<std::string>();
            cameras[n_cameras-1].maskFileOrientPath = cameras_yaml[i]["maskFileOrient"].as<std::string>();
            cameras[n_cameras-1].show               = false;
        }
    }

    for(auto c: cameras)
        std::cout<<c<<std::endl;

    return true;
}



int main(int argc, char **argv)
{
    std::vector<edge::camera> cameras;
    std::string net = "yolo3_berkeley_fp32.rt";
    char type = 'y';
    int n_classes = 80;
    readParameters(argc, argv, cameras, net, type, n_classes);
    initializeCamerasNetworks(cameras, net, type, n_classes);

    return EXIT_SUCCESS;
}
