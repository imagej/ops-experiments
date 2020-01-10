find_package(OpenCL)



# find clFFT path
FIND_PATH(CLFFT_LIBRARY_DIR $ENV{CLFFT_LIBRARY_DIR} [DOC "CLFFT library path"])


link_directories(../../native/lib/ ${CLFFT_LIBRARY_DIR})
add_library(opencldeconv SHARED opencldeconv.cpp)
target_link_libraries(opencldeconv clFFT ${OpenCL_LIBRARY})

install(TARGETS opencldeconv DESTINATION lib)
