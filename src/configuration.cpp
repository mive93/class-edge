#include "configuration.h"

std::ostream& operator<<(std::ostream& os, const edge::camera& c){
    os << c.id << '\t' << c.input << '\t' << c.pmatrixPath<< '\t' << c.cameraCalibPath;
    return os;
}

std::string executeCommandAndGetOutput(const char * command)
{
    FILE *fpipe;
    char c = 0;

    if (0 == (fpipe = (FILE*)popen(command, "r")))
        FatalError("popen() failed.");

    std::string output; 
    while (fread(&c, sizeof c, 1, fpipe))
        output.push_back(c);
    // std::cout<<output<<std::endl;

    pclose(fpipe);
    return output;
}

std::string decryptString(std::string encrypted, const std::string& password){
    std::string command = "echo '"+ encrypted +"' | openssl enc -e -aes-256-cbc -a -d -salt -iter 100000 -pass pass:"+password;
    return executeCommandAndGetOutput(command.c_str());
}

std::string encryptString(std::string to_encrypt, const std::string& password){
    std::string command = "echo -n "+to_encrypt+" | openssl enc -e -aes-256-cbc -a -salt -iter 100000 -pass pass:"+password;
    return executeCommandAndGetOutput(command.c_str());
}

void readParamsFromYaml(const std::string& params_path, const std::vector<int>& cameras_ids,std::vector<edge::camera>& cameras,std::string& net, char& type, int& n_classes, std::string& tif_map_path)
{
    std::string password = ""; 
    YAML::Node config   = YAML::LoadFile(params_path);
    net          = config["net"].as<std::string>();
    type         = config["type"].as<char>();
    n_classes    = config["classes"].as<int>();
    tif_map_path = config["tif"].as<std::string>();
    
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
        {
            if(password == "") {
                std::cout<<"Please insert the password to decript the cameras input"<<std::endl;
                std::cin>>password;
            }
            cameras[n_cameras-1].input          = decryptString(cameras_yaml[i]["input"].as<std::string>(), password);
        }
        else            
            cameras[n_cameras-1].input          = cameras_yaml[i]["input"].as<std::string>();
        cameras[n_cameras-1].pmatrixPath        = cameras_yaml[i]["pmatrix"].as<std::string>();
        cameras[n_cameras-1].maskfilePath       = cameras_yaml[i]["maskfile"].as<std::string>();
        cameras[n_cameras-1].cameraCalibPath    = cameras_yaml[i]["cameraCalib"].as<std::string>();
        cameras[n_cameras-1].maskFileOrientPath = cameras_yaml[i]["maskFileOrient"].as<std::string>();
        cameras[n_cameras-1].show               = false;
    }
}

bool readParameters(int argc, char **argv,std:: vector<edge::camera>& cameras,std::string& net, char& type, int& n_classes, std::string& tif_map_path){
    std::string help =  "class-edge demo\nCommand:\n"
                        "-i\tparameters file\n"
                        "-n\tnetwork rt path\n"
                        "-t\ttype of network (only y|c|m admissed)\n"
                        "-c\tnumber of classes for the network\n"
                        "-m\tmap.tif path (to get GPS position)\n"
                        "\tlist of camera ids (n ids expected)\n\n";

    //default values
    net             = "yolo3_berkeley_fp32.rt";
    tif_map_path    = "../data/masa_map.tif";
    type            = 'y';
    n_classes       = 80;

    //read values
    std::string params_path         = "";
    std::string read_net            = "";
    std::string read_tif_map_path   = "";
    char read_type                  = '\0';
    int read_n_classes              = 0;
    
    //read arguments
    for(int opt;(opt = getopt(argc, argv, ":i:m:n:c:t:h")) != -1;)
    {
        switch(opt)
        {
            case 'h':
                std::cout<<help<<std::endl;
                return false;
            case 'i':
                params_path = optarg;
                break;
            case 'm':
                read_tif_map_path = optarg;
                break;
            case 'c':
                read_n_classes = atoi(optarg);
                break;
            case 'n':
                read_net = optarg;
                break;
            case 't':
                read_type = optarg[0];
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

    std::cout<<cameras_ids.size()<<" camera id(s) given: ";
    for(size_t i=0; i < cameras_ids.size(); ++i)
        std::cout<<cameras_ids[i]<<" ";
    std::cout<<std::endl;

    //if no parameters file given, set all default values for 1 camera
    if(params_path == "") {
        cameras.resize(1);
        cameras[0].id                   = 20936;
        cameras[0].input                = "../data/20936.mp4";
        cameras[0].pmatrixPath          = "../data/pmat_new/pmat_07-03-20936_20p.txt";
        cameras[0].maskfilePath         = "../data/masks/20936_mask.jpg";
        cameras[0].cameraCalibPath      = "../data/calib_cameras/20936.params";
        cameras[0].maskFileOrientPath   = "../data/masks_orient/1920-1080_mask_null.jpg";
        cameras[0].show                 = false;
    }
    else 
        readParamsFromYaml(params_path, cameras_ids, cameras, net, type, n_classes, tif_map_path);

    //if specified from command line, override parameters read from file 
    if (read_net != "")             net             = read_net;
    if (read_tif_map_path != "")    tif_map_path    = read_tif_map_path;
    if (read_type != '\0')          type            = read_type;
    if (read_n_classes != 0)        n_classes       = read_n_classes;

    std::cout<<"Input parameters file in use:\t"<<params_path<<std::endl;
    std::cout<<"Tif map in use:\t\t\t"<<tif_map_path<<std::endl;
    std::cout<<"Network rt to use:\t\t"<<net<<std::endl;
    std::cout<<"Type of network in use:\t\t"<<type<<std::endl;
    std::cout<<"Number of classes specified:\t"<<n_classes<<std::endl;

    return true;
}

void initializeCamerasNetworks(std:: vector<edge::camera>& cameras, const std::string& net, const char type, int& n_classes){
    //if the rt file does not esits, run the test to create it
    if(!fileExist(net.c_str()))
    {
        std::string test_cmd = "tkDNN/test_" + net.substr(0, net.find("_fp"));
        if(!fileExist(test_cmd.c_str()))
            FatalError("Wrong network, the test does not exist for tkDNN");
        system(test_cmd.c_str());        
    }

    //assign to each camera a detector
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