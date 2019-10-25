cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(YacuDecu LANGUAGES CXX CUDA)

set(PROJECT_SRC ../../YacuDecu/src/deconv.cu)

add_library(YacuDecu SHARED ${PROJECT_SRC})
#cuda_add_cufft_to_target(YacuDecu)

target_link_libraries(YacuDecu cufft)

install(TARGETS YacuDecu DESTINATION lib)

