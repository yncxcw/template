cmake_minimum_required (VERSION 3.13)

project(CUDA_EXAMPKE LANGUAGES CXX CUDA)
include_directories(include)


add_library(cuda_kernel_lib kernel.cu)
target_compile_features(cuda_kernel_lib PRIVATE cxx_std_17)

add_executable(main main.cc)
target_link_libraries(main cuda_kernel_lib)