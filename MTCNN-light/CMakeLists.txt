set(DEMO_NAME light)

AUX_SOURCE_DIRECTORY(src SRC_FILES)
add_executable(${DEMO_NAME} ${SRC_FILES})

set(DEPS ${OpenCV_LIBS})
find_package(BLAS REQUIRED)
list(APPEND DEPS ${BLAS_LIBRARIES})
target_link_libraries(${DEMO_NAME} ${DEPS})