cd ~/Downloads
wget https://github.com/glfw/glfw/releases/download/3.3/glfw-3.3.zip
unzip glfw-3.3.zip
cd glfw-3.3/
mkdir build
cd build
cmake ..
make -j4
sudo make install
