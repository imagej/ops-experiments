# this project is used to wrap mkl deconvolution code
project(MKLFFTW)

# find MKL include
FIND_PATH( MKL_INCLUDE_DIR $ENV{MKL_INCLUDE_DIR} [DOC "MKl include path"])

# find MKL lib
FIND_PATH( MKL_LIBRARY_DIR $ENV{MKL_LIBRARY_DIR} [DOC "MKl library path"])

# find Open MP library
FIND_PATH( OMP_LIBRARY_DIR $ENV{OMP_LIBRARY_DIR} [DOC "OPM library path"])

include_directories(source ${MKL_INCLUDE_DIR}/)
link_directories(${MKL_LIBRARY_DIR} ${OMP_LIBRARY_DIR}/)

add_library(MKLFFTW source/MKLFFTW.cpp)

target_link_libraries(MKLFFTW mkl_core mkl_intel_lp64 mkl_intel_thread mkl_tbb_thread mkl_rt libiomp5md.lib) 
