# this project is used to wrap YacuDecu cuda deconvolution by Bob Pepin
project(YacuDecuWrapperProject)

# Find Cuda
find_package(CUDA REQUIRED)

# add the cuda library
cuda_add_library(YacuDecu deconv.cu)
CUDA_ADD_CUFFT_TO_TARGET( YacuDecu )

set(LIB_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib)

INSTALL(TARGETS YacuDecu DESTINATION ${LIB_PATH})

