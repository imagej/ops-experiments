find_package(ArrayFire)

if(ArrayFire_CPU_FOUND)
  # Hello World example
  add_library(arrayfiredecon_cpu SHARED arrayfiredecon.cpp) 
  target_link_libraries(arrayfiredecon_cpu ArrayFire::afcpu)
  install(TARGETS arrayfiredecon_cpu DESTINATION lib)
endif()

if(ArrayFire_CUDA_FOUND)
  add_library(arrayfiredecon_cuda SHARED arrayfiredecon.cpp) 
    target_link_libraries(arrayfiredecon_cuda ArrayFire::afcuda)
endif()

if(ArrayFire_OpenCL_FOUND)
    add_library(arrayfiredecon_opencl SHARED arrayfiredecon.cpp) 
    target_link_libraries(arrayfiredecon_opencl ArrayFire::afopencl)
endif()

add_library(arrayfiredecon SHARED arrayfiredecon.cpp) 

# To use Unified backend, do the following.
# Unified backend lets you choose the backend at runtime
target_link_libraries(arrayfiredecon ArrayFire::af)


