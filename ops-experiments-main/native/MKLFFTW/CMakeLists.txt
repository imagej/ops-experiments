# this project is used to wrap mkl deconvolution code
project(MKLFFTW)

# find MKL include
FIND_PATH( MKL_INCLUDE_DIR $ENV{MKL_INCLUDE_DIR} [DOC "MKl include path"])

# find MKL lib
FIND_PATH( MKL_LIBRARY_DIR $ENV{MKL_LIBRARY_DIR} [DOC "MKl library path"])

# find Open MP library
FIND_PATH( OMP_LIBRARY_DIR $ENV{OMP_LIBRARY_DIR} [DOC "OPM library path"])

include_directories(${MKL_INCLUDE_DIR}/)
link_directories(${MKL_LIBRARY_DIR} ${OMP_LIBRARY_DIR}/)

add_library(MKLFFTW src/MKLFFTW.cpp)

set_property(TARGET MKLFFTW PROPERTY POSITION_INDEPENDENT_CODE ON)

#target_link_libraries(MKLFFTW mkl_intel_lp64 mkl_intel_thread mkl_core mkl_avx2 mkl_def iomp5) 
target_link_libraries(MKLFFTW mkl_rt pthread m dl) 

install(TARGETS MKLFFTW DESTINATION lib)
