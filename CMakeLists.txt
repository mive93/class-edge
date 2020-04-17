cmake_minimum_required(VERSION 3.5)
project(class-edge CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wno-deprecated-declarations -Wno-unused-variable")

#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
find_package(CUDA 9.0 REQUIRED)
SET(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 --compiler-options '-fPIC'")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")

find_package(yaml-cpp REQUIRED)

#-------------------------------------------------------------------------------
# Build libraries
#-------------------------------------------------------------------------------


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include 
                    ${CMAKE_CURRENT_SOURCE_DIR}/tracker_CLASS/c++/include 
                    ${CMAKE_CURRENT_SOURCE_DIR}/tkDNN/include/tkDNN 
                    ${CMAKE_CURRENT_SOURCE_DIR}/tkCommon/include/tkCommon
                    ${CMAKE_CURRENT_SOURCE_DIR}/masa_protocol/include)
include_directories(${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${NVINFER_INCLUDES} ${CUDNN_INCLUDE_DIRS})

file(GLOB calss-edge-SRC "src/*.cpp")
set(class-edge-LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)
add_library(class-edge SHARED ${calss-edge-SRC})
target_link_libraries(class-edge ${class-edge-LIBS} )

#-------------------------------------------------------------------------------
# Build executables
#-------------------------------------------------------------------------------
add_executable(main src/main.cpp)
target_link_libraries(main class-edge)
