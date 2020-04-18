#include <iostream>
#include "configuration.h"

int main(int argc, char **argv)
{
    if(argc < 3)
        FatalError("parameters file required (input first, output second)");
    std::string params_path_in = argv[1]; 
    std::string params_path_out = argv[2]; 
    
    std::string password;
    std::cout<<"Insert a password to encrypt the cameras input"<<std::endl;
    std::cin>>password;

    YAML::Node config = YAML::LoadFile(params_path_in);    
    std::string input;
    for(int i=0; i<config["cameras"].size(); i++)
    {
        //if encrypted is set to 0, then ecrypt
        if (!config["cameras"][i]["encrypted"].as<int>())
        {
            std::cout<<"Convert: "<<config["cameras"][i]["input"].as<std::string>()<<std::endl;
            input = encryptString(config["cameras"][i]["input"].as<std::string>(), password);
            std::cout<<"into: "<<input<<std::endl;
            config["cameras"][i]["encrypted"] = "1";
            config["cameras"][i]["input"] = input;
        }
    }

    //write new yaml
    std::ofstream fout(params_path_out);
    fout << config;
    fout.close();

    return EXIT_SUCCESS;
}
