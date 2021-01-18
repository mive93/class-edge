#include "configuration.h"


std::string executeCommandAndGetOutput(const char * command){
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
    // when using OpenSSL 1.1.1 use the following
    // std::string command = "echo '"+ encrypted +"' | openssl enc -e -aes-256-cbc -a -d -salt -iter 100000 -pass pass:"+password;
    std::string command = "echo '"+ encrypted +"' | openssl enc -e -aes-256-cbc -a -d -salt -pass pass:"+password;
    return executeCommandAndGetOutput(command.c_str());
}

std::string encryptString(std::string to_encrypt, const std::string& password){
    // when using OpenSSL 1.1.1 use the following
    // std::string command = "echo -n "+to_encrypt+" | openssl enc -e -aes-256-cbc -a -salt -iter 100000 -pass pass:"+password;
    std::string command = "echo -n "+to_encrypt+" | openssl enc -e -aes-256-cbc -a -salt -pass pass:"+password;
    return executeCommandAndGetOutput(command.c_str());
}

void readParamsFromYaml(const std::string& params_path, const std::vector<int>& cameras_ids,std::vector<edge::camera_params>& cameras_par,std::string& net, char& type, int& n_classes, std::string& tif_map_path){
    std::string password = ""; 
    YAML::Node config   = YAML::LoadFile(params_path);
    net             = config["net"].as<std::string>();
    type            = config["type"].as<char>();
    n_classes       = config["classes"].as<int>();
    tif_map_path    = config["tif"].as<std::string>();
    if(config["password"])
        password    = config["password"].as<std::string>();


    int stream_height, stream_width;
    stream_width    = config["width"].as<int>();
    stream_height   = config["height"].as<int>();

    YAML::Node cameras_yaml = config["cameras"];
    bool use_info;
    int n_cameras = 0;
    for(int i=0; i<cameras_yaml.size(); i++){
        int camera_id = cameras_yaml[i]["id"].as<int>();

        //save infos only of the cameras whose ids where passed as args
        use_info = false;
        for(auto id: cameras_ids)
            if(camera_id == id)
                use_info = true;
        if(!use_info) continue;

        cameras_par.resize(++n_cameras);
        cameras_par[n_cameras-1].id                 = camera_id;
        if (cameras_yaml[i]["encrypted"].as<int>()){
            if(password == "") {
                std::cout<<"Please insert the password to decript the cameras input"<<std::endl;
                std::cin>>password;
            }
            cameras_par[n_cameras-1].input          = decryptString(cameras_yaml[i]["input"].as<std::string>(), password);
        }
        else            
            cameras_par[n_cameras-1].input          = cameras_yaml[i]["input"].as<std::string>();
        cameras_par[n_cameras-1].pmatrixPath        = cameras_yaml[i]["pmatrix"].as<std::string>();
        cameras_par[n_cameras-1].maskfilePath       = cameras_yaml[i]["maskfile"].as<std::string>();
        cameras_par[n_cameras-1].cameraCalibPath    = cameras_yaml[i]["cameraCalib"].as<std::string>();
        cameras_par[n_cameras-1].maskFileOrientPath = cameras_yaml[i]["maskFileOrient"].as<std::string>();
        cameras_par[n_cameras-1].streamWidth        = stream_width;
        cameras_par[n_cameras-1].streamHeight       = stream_height;
        cameras_par[n_cameras-1].show               = true;
    }
}

bool readParameters(int argc, char **argv,std:: vector<edge::camera_params>& cameras_par,std::string& net, char& type, int& n_classes, std::string& tif_map_path){
    std::string help =  "class-edge demo\nCommand:\n"
                        "-i\tparameters file\n"
                        "-n\tnetwork rt path\n"
                        "-t\ttype of network (only y|c|m admissed)\n"
                        "-c\tnumber of classes for the network\n"
                        "-m\tmap.tif path (to get GPS position)\n"
                        "-s\tshow (0=false, 1=true)\n"
                        "-v\tverbose (0=false, 1=true)\n"
                        "\tlist of camera ids (n ids expected)\n\n";

    //default values
    net             = "yolo3_berkeley_fp32.rt";
    tif_map_path    = "../data/masa_map.tif";
    type            = 'y';
    n_classes       = 10;

    //read values
    std::string params_path         = "";
    std::string read_net            = "";
    std::string read_tif_map_path   = "";
    char read_type                  = '\0';
    int read_n_classes              = 0;
    
    //read arguments
    for(int opt;(opt = getopt(argc, argv, ":i:m:s:v:n:c:t:h")) != -1;){
        switch(opt){
            case 'h':
                std::cout<<help<<std::endl;
                exit(EXIT_SUCCESS);
            case 'i':
                params_path = optarg;
                break;
            case 'm':
                read_tif_map_path = optarg;
                break;
            case 'c':
                read_n_classes = atoi(optarg);
                break;
            case 's':
                show = atoi(optarg);
                break;
            case 'v':
                verbose = atoi(optarg);
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
        cameras_par.resize(1);
        cameras_par[0].id                   = 20936;
        cameras_par[0].input                = "../data/20936.mp4";
        cameras_par[0].pmatrixPath          = "../data/pmat_new/pmat_07-03-20936_20p.txt";
        cameras_par[0].maskfilePath         = "../data/masks/20936_mask.jpg";
        cameras_par[0].cameraCalibPath      = "../data/calib_cameras/20936.params";
        cameras_par[0].maskFileOrientPath   = "../data/masks_orient/1920-1080_mask_null.jpg";
        cameras_par[0].streamWidth          = 960;
        cameras_par[0].streamHeight         = 540;
        cameras_par[0].show                 = true;
    }
    else 
        readParamsFromYaml(params_path, cameras_ids, cameras_par, net, type, n_classes, tif_map_path);

    //if specified from command line, override parameters read from file 
    if (read_net != "")             net             = read_net;
    if (read_tif_map_path != "")    tif_map_path    = read_tif_map_path;
    if (read_type != '\0')          type            = read_type;
    if (read_n_classes != 0)        n_classes       = read_n_classes;

    std::cout<<"Input parameters file in use:\t"<<params_path<<std::endl;
    std::cout<<"Tif map in use:\t\t\t"<<tif_map_path<<std::endl;
    std::cout<<"Network rt to use:\t\t"<<net<<std::endl;
    std::cout<<"Type of network in use:\t\t"<<type<<std::endl;
    std::cout<<"Number of classes specified:\t"<<n_classes<<std::endl<<std::endl;

    return true;
}

void initializeCamerasNetworks(std:: vector<edge::camera>& cameras, const std::string& net, const char type, int& n_classes){
    //if the rt file does not esits, run the test to create it
    if(!fileExist(net.c_str())){
        std::string test_cmd = "tkDNN/test_" + net.substr(0, net.find("_fp"));
        std::string precision =  net.substr(net.find("_fp")+3, 2);
        if(std::stoi(precision) == 16)
            setenv("TKDNN_MODE","FP16",1);

        if(!fileExist(test_cmd.c_str()))
            FatalError("Wrong network, the test does not exist for tkDNN");
        system(test_cmd.c_str());        
    }

    if(!fileExist(net.c_str()))
        FatalError("Problem with rt creation");

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


void readProjectionMatrix(const std::string& path, cv::Mat& prj_mat)
{
    std::ifstream prj_mat_file;
    prj_mat_file.open(path);

    prj_mat = cv::Mat(cv::Size(3, 3), CV_64FC1);
    double *vals = (double *)prj_mat.data;
    
    double number = 0 ;      
    int i;
    for(i=0; prj_mat_file >> number && i<prj_mat.cols*prj_mat.rows ; ++i)
        vals[i] = number;

    prj_mat_file.close();

    if (i != prj_mat.cols*prj_mat.rows)
        FatalError("Problem with projection matrix file");    
}

void readCalibrationMatrix(const std::string& path, cv::Mat& calib_mat, cv::Mat& dist_coeff, int& image_width, int& image_height)
{
    YAML::Node config   = YAML::LoadFile(path);

    //read calibration size
    image_width     = config["image_width"].as<int>();
    image_height    = config["image_height"].as<int>();

    //read camera matrix
    int rows = config["camera_matrix"]["rows"].as<int>();
    int cols = config["camera_matrix"]["cols"].as<int>();

    cv::Mat calib = cv::Mat(cv::Size(rows, cols), CV_64FC1);
    double *vals = (double *)calib.data;
    
    for(int i=0; i < config["camera_matrix"]["data"].size(); ++i )
        vals[i] = config["camera_matrix"]["data"][i].as<double>();

    calib_mat = calib;

    //read distortion coefficents
    rows = config["distortion_coefficients"]["rows"].as<int>();
    cols = config["distortion_coefficients"]["cols"].as<int>();

    cv::Mat coeff = cv::Mat(cv::Size(rows, cols), CV_64FC1);
    vals = (double *)coeff.data;

    for(int i=0; i < config["distortion_coefficients"]["data"].size(); ++i )
        vals[i] = config["distortion_coefficients"]["data"][i].as<double>();

    dist_coeff = coeff;
}

void readTiff(const std::string& path, double *adfGeoTransform)
{
    if(!fileExist(path.c_str())){
        std::cout<<path<<std::endl;
        if(path == "../data/masa_map.tif")
            system("wget https://cloud.hipert.unimore.it/s/rWrdd2ygKF48eGp/download -O ../data/masa_map.tif");
        if(!fileExist(path.c_str()))
            FatalError("Tif map given does not exit. Needed for tracker.");
    }

    GDALDataset *poDataset;
    GDALAllRegister();
    poDataset = (GDALDataset *)GDALOpen(path.c_str(), GA_ReadOnly);
    if (poDataset != NULL)
    {
        poDataset->GetGeoTransform(adfGeoTransform);
    }
}

void readCaches(edge::camera& cam){
    std::string error_mat_data_path = "../data/cache/" + std::to_string(cam.id) + "/Error_Matrix";
    cam.precision = cv::Mat(cv::Size(cam.calibWidth, cam.calibHeight), CV_32F, 0.0); 

    std::ifstream error_mat;
    error_mat.open(error_mat_data_path.c_str());
    if (!error_mat){
        system("wget https://cloud.hipert.unimore.it/s/MSXGPBjqsDf8goo/download -O ../data/camera_caches.zip");
        system("unzip -d ../data/ ../data/camera_caches.zip");
        system("rm ../data/camera_caches.zip");
        
        error_mat.open(error_mat_data_path.c_str());
        if (!error_mat)
            FatalError("Could not find caches file for camera nor download it");
    }

    for (int y = 0; y < cam.calibHeight; y++){ //height (number of rows)
        for (int x = 0; x < cam.calibWidth; x++) { //width (number of columns)
            float tmp;
            error_mat.read(reinterpret_cast<char*> (&tmp), sizeof(float));
            cam.precision.at<float>(y,x) = tmp;
        }
    }  
}

std::vector<edge::camera> configure(int argc, char **argv)
{
    std::vector<edge::camera_params> cameras_par;
    std::string net, tif_map_path;
    char type;
    int n_classes;

    //read args from command line
    readParameters(argc, argv, cameras_par, net, type, n_classes, tif_map_path);

    //set dataset
    edge::Dataset_t dataset;
    switch(n_classes){
        case 10: dataset = edge::Dataset_t::BDD; break;
        case 80: dataset = edge::Dataset_t::COCO; break;
        default: FatalError("Dataset type not supported yet, check number of classes in parameter file.");
    }
    
    if(verbose){
        for(auto cp: cameras_par)
            std::cout<<cp;
    }

    //read calibration matrixes for each camera
    std::vector<edge::camera> cameras(cameras_par.size());
    for(size_t i=0; i<cameras.size(); ++i){
        readCalibrationMatrix(cameras_par[i].cameraCalibPath, cameras[i].calibMat, cameras[i].distCoeff, cameras[i].calibWidth, cameras[i].calibHeight);
        readProjectionMatrix(cameras_par[i].pmatrixPath, cameras[i].prjMat);
        cameras[i].id           = cameras_par[i].id;
        cameras[i].input        = cameras_par[i].input;
        cameras[i].streamWidth  = cameras_par[i].streamWidth;
        cameras[i].streamHeight = cameras_par[i].streamHeight;
        cameras[i].show         = cameras_par[i].show;
        cameras[i].invPrjMat    = cameras[i].prjMat.inv();
        cameras[i].dataset      = dataset;
    }

    //initialize neural netwokr for each camera
    initializeCamerasNetworks(cameras, net, type, n_classes);

    if(verbose){
        for(auto c: cameras)
            std::cout<<c;
    }

    //read tif image to get georeference parameters
    double* adfGeoTransform = (double *)malloc(6 * sizeof(double));
    readTiff(tif_map_path, adfGeoTransform);
    if(verbose){
        for(int i=0; i<6; i++)
            std::cout<<adfGeoTransform[i]<<" ";
        std::cout<<std::endl;
    }
    
    for(auto& c: cameras)
    {
        readCaches(c);

        c.adfGeoTransform = (double *)malloc(6 * sizeof(double));
        memcpy(c.adfGeoTransform, adfGeoTransform, 6 * sizeof(double) );

        //initialize the geodetic converter with a point in the MASA
        c.geoConv.initialiseReference(44.655540, 10.934315, 0);
    }
    free(adfGeoTransform);    

    std::cout<<COL_ORANGEB<<"class-edge version "<<VERSION_MAJOR<<"."<<VERSION_MINOR<< COL_END<<std::endl;

    return cameras;
}