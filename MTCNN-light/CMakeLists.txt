set(DEMO_NAME light)

AUX_SOURCE_DIRECTORY(src SRC_FILES)
add_executable(${DEMO_NAME} ${SRC_FILES})

find_package(OpenBLAS REQUIRED)
MESSAGE(INFO ${OpenBLAS_LIB})
target_link_libraries(${DEMO_NAME} ${OpenCV_LIBS} ${OpenBLAS_LIB})
file(COPY model DESTINATION ${CMAKE_BINARY_DIR}/MTCNN-light)