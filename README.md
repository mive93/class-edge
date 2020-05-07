# class-edge

class-edge provides the software for the edges of a smart city, i.e. smart cameras, in the context of the European Project CLASS (H2020, G.A. 780622)

## Dependencies

This projects depends on: 

  * CUDA 10.0
  * CUDNN 7.603
  * TENSORRT 6.01
  * OPENCV 3.4
  * yaml-cpp 0.5.2 
  * Eigen
  * GDal

```
sudo apt-get install -y libeigen3-dev \
                        python3-matplotlib \
                        python-dev \
                        libgdal-dev \
                        libcereal-dev \
                        libyaml-cpp-dev
```

required for tkCommon
```
sudo apt-get install -y libgles2-mesa-dev libglew-dev libmatio-dev libpcap-dev
bash scripts/install-glfw-3.3.sh
```

## How to build the project

```
https://git.hipert.unimore.it/mverucchi/class-edge.git
cd class-edge
git submodule update --init --recursive 
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j
```

## How to lunch the edge

In general (use ./edge -h for help)
```
./edge -i <parameter-file> <cam-id-1> <cam-id-2> ... <cam-id-8> 
```
Example:
```
./edge -i ../data/all_cameras_en.yaml 20939 20940 20936 6310 634
```

## How to initialize or update submodule
```
git submodule update --init --recursive #initialize
git submodule update --remote --recursive  #update all
```

## How to encrypt 

This is how you encrypt a string (omit -iter 100000 with Ubuntu 16.04):
```
echo -n "yourAwesomeString" | openssl enc -e -aes-256-cbc -a -salt -iter 100000
```
In you want to encrypt the input of a parameters file, be sure that the field ```encrypted``` is set to 0.
Then just run 
```
./encrypt <params-no-enc> <params-enc>
```
where
  * ```<params-no-enc>``` is the input parametrs file (yaml)
  * ```<params-enc>``` is the output parametrs file (yaml) with all the input encrypted with the password the program will ask for.


## How to remove tkDNN prints
Go to tkDNN/include/tkDNN/DetectionNN.h and change the update function with this: 
```
          void update(cv::Mat &frame, bool save_times=false, std::ofstream *times=nullptr){
            if(!frame.data)
                FatalError("No image data feed to detection");

            if(save_times && times==nullptr)
                FatalError("save_times set to true, but no valid ofstream given");

            originalSize = frame.size();
            // printCenteredTitle(" TENSORRT detection ", '=', 30); 
            {
                // TIMER_START
                preprocess(frame);
                // TIMER_STOP
                // if(save_times) *times<<t_ns<<";";
            }

            //do inference
            tk::dnn::dataDim_t dim = netRT->input_dim;
            {
                // dim.print();
                // TIMER_START
                netRT->infer(dim, input_d);
                // TIMER_STOP
                // dim.print();
                // stats.push_back(t_ns);
                // if(save_times) *times<<t_ns<<";";
            }

            {
                // TIMER_START
                postprocess();
                // TIMER_STOP?
                // if(save_times) *times<<t_ns<<"\n";
            }
        }  
```