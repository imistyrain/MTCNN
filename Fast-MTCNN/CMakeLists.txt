AUX_SOURCE_DIRECTORY(./ SRC_FILES)
set(DEMO_NAME Fast-MTCNN)
add_executable(${DEMO_NAME} ${SRC_FILES})

target_link_libraries(${DEMO_NAME} ${OpenCV_LIBS})