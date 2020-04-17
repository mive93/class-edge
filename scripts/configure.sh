mkdir build
git submodule update --init --recursive
git submodule update --remote --recursive   #remove for deploy
mkdir tkDNN/build
cd build
cmake ..
make -j
cd tkDNN/build
./test_yolo3_berkeley
mv yolo3_berkeley_fp32.rt ../../build
cd ../../build