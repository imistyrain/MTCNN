cmake_minimum_required(VERSION 3.0)

project(MTCNN)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3.4 REQUIRED)
find_package(Boost COMPONENTS system thread filesystem python REQUIRED)
find_package(Caffe)
if(Caffe_LIBRARIES)
	include_directories(${Caffe_INCLUDE_DIRS})
	add_subdirectory(caffe)
else()
	message("Caffe not found, cpp was not built")
endif()

find_package(ncnn)
if(ncnn_FOUND)
	add_subdirectory(ncnn)
else()
	message("ncnn not found, ntcnn was not built")
endif()

#default build light, only depend on openblas
add_subdirectory(MTCNN-light)

#need opencv version >= 3.4.* or recompile opencv with conrtib module
add_subdirectory(cpp)

file(GLOB models RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "./model/*")
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/model/)
file(COPY ${models} DESTINATION ${CMAKE_BINARY_DIR}/model/)

file(GLOB images RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "./images/*")
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/images/)
file(COPY ${images} DESTINATION ${CMAKE_BINARY_DIR}/images/)