cmake_minimum_required(VERSION 2.8)

project(MTCNN)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
option(BUILD_CAFFE "build with caffe" ON)
option(BUILD_NCNN "build with ncnn" ON)

#default build light, only depend on openblas
add_subdirectory(MTCNN-light)

#need opencv version >= 3.4.* or recompile opencv with conrtib module
#add_subdirectory(Fast-MTCNN)

if(BUILD_CAFFE)
	find_package(Caffe  REQUIRED)
	include_directories(${Caffe_INCLUDE_DIRS})
	add_subdirectory(cpp)
endif()

if(BUILD_NCNN)
	add_subdirectory(ncnn)
endif()