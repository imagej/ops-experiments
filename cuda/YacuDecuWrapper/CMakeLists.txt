# this project is used to wrap YacuDecu cuda deconvolution by Bob Pepin
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(YacuDecuWrapperProject LANGUAGES C CUDA)

# add the cuda library
add_library(YacuDecu deconv.cu)
#CUDA_ADD_CUFFT_TO_TARGET( YacuDecu )

#set_property(TARGET YacuDecu
#             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET YacuDecu PROPERTY POSITION_INDEPENDENT_CODE ON) 

set(LIB_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib)

INSTALL(TARGETS YacuDecu DESTINATION ${LIB_PATH})

