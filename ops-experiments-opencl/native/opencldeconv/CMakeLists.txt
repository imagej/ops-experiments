find_package(OpenCL)

link_directories(../../native/lib/)
add_library(opencldeconv SHARED opencldeconv.cpp)
target_link_libraries(opencldeconv clFFT ${OpenCL_LIBRARY})
