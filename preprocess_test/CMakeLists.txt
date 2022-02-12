cmake_minimum_required(VERSION 2.6)

project(preprocess)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
#option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
#cuda_add_library(myplugins SHARED yololayer.cu)
#target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

#cuda_add_executable(run yolov5.cpp preprocess.cu)
cuda_add_library(preprocess STATIC preprocess.cu preprocess.h cuda_utils.h)

#target_link_libraries(yolov5 nvinfer)
#target_link_libraries(run cudart)
##target_link_libraries(preprocess cudart)
#target_link_libraries(yolov5 myplugins)
##target_link_libraries(preprocess ${OpenCV_LIBS})
##target_compile_features(preprocess PUBLIC)
if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)


