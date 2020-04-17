mkdir build
git submodule update --init --recursive
git submodule update --remote     #remove for deploy
git pull --recurse-submodules     #remove for deploy
cd tkDNN
mkdir build
cd build
cmake ..
make -j
./test_yolo3_berkeley
mv yolo3_berkeley_fp32.rt ../../build
cd ../../build
cmake ..
make -j