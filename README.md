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
git clone gitlab@git.hipert.unimore.it:mverucchi/class-edge.git
cd class-edge
mkdir build
cd build
cmake ..
make -j
```

## How to initialize or update submodule
```
git submodule update --init --recursive #initialize
git submodule update --remote --recursive  #update all
```

## How to encrypt a string

```
echo -n "yourAwesomeString" | openssl enc -e -aes-256-cbc -a -salt -iter 100000
```