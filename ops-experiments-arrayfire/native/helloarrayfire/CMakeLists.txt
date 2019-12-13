find_package(ArrayFire)
add_executable(helloarrayfire helloarrayfire.cpp) 

# To use Unified backend, do the following.
# Unified backend lets you choose the backend at runtime
target_link_libraries(helloarrayfire ArrayFire::af)
